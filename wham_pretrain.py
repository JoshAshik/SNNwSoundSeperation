"""
wham_pretrain.py — Pre-train SNN-TasNet encoder/decoder on WHAM!

This script trains the model in GRU mode on WHAM!'s 20,000 two-speaker
separation examples. The goal is NOT to get a perfect WHAM! separator —
it's to train the encoder and decoder to understand audio separation at all,
using a large dataset before fine-tuning on our 300 scenes.

After this completes, run snn_finetune.py which:
  1. Loads this checkpoint
  2. Swaps the GRU separator for a ChunkedSNNSeparator
  3. Freezes encoder/decoder for the first N epochs
  4. Fine-tunes everything on the 400-scene corpus

Usage:
    # Step 1: download WHAM!
    python wham_dataset.py --download --wham_root ./wham_data

    # Step 2: verify
    python wham_dataset.py --verify --wham_root ./wham_data

    # Step 3: pre-train (~6-8 hours on GTX 1660 Super)
    python wham_pretrain.py --wham_root ./wham_data

    # Step 4: fine-tune
    python snn_finetune.py --pretrain_ckpt ./checkpoints_pretrain/best_pretrain.pt

Training strategy:
    - 100 epochs (WHAM! has enough data that convergence is much faster)
    - Larger batch size (8) — WHAM! clips are slightly longer than 3s but
      still fit comfortably in 6GB VRAM
    - Cosine annealing T0=30 (shorter runs benefit from earlier restarts)
    - Same SI-SDR + spectral loss as fine-tuning
    - EMA weights saved to checkpoint for stable transfer
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from sep_losses  import SeparationLoss
from sep_metrics import fast_si_sdri
from sep_model   import SNNTasNet, count_parameters
from wham_dataset import build_wham_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# EMA (same as sep_train.py)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model   = model
        self.decay   = decay
        self.shadow  = {n: p.data.clone() for n, p in model.named_parameters()}
        self._backup = {}

    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        self._backup = {n: p.data.clone() for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self._backup[n])

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, sd):
        self.shadow = sd["shadow"]
        self.decay  = sd["decay"]


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    wham_root      = "./wham_data",
    n_epochs       = 100,
    batch_size     = 8,
    lr             = 3e-4,
    weight_decay   = 1e-5,
    grad_clip      = 5.0,
    n_filters      = 256,
    kernel_sz      = 32,
    stride         = 16,
    hidden         = 192,
    n_layers       = 2,
    dropout        = 0.2,        # slightly less dropout for large dataset
    lambda_spec    = 0.5,
    chunk_size     = 250,
    cosine_t0      = 30,
    cosine_tmult   = 2,
    ema_decay      = 0.999,
    checkpoint_dir = "./checkpoints_pretrain",
    log_dir        = "./runs_pretrain",
    csv_path       = "./pretrain_log.csv",
    checkpoint_every = 10,
    device         = "cuda",
    num_workers    = 0,
    seed           = 42,
    resume         = False,
    max_train      = None,       # set e.g. 5000 for a quick smoke test
)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loss_fn, loader, optimizer, device, scaler,
              grad_clip, is_train, ema=None, warmup_sched=None):
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_sisdr = total_spec = 0.0
    n_batches  = 0

    with ctx:
        for batch in loader:
            mixture = batch["mixture"].to(device, non_blocking=True)
            sources = batch["sources"].to(device, non_blocking=True)

            with autocast("cuda", enabled=(scaler is not None)):
                out    = model(mixture)
                losses = loss_fn(out["separated"], sources, out["spike_recs"])

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if ema is not None:
                    ema.update()
                if warmup_sched is not None:
                    warmup_sched.step()

            with torch.no_grad():
                sisdr_i = fast_si_sdri(
                    out["separated"].float(), sources.float(), mixture.float()
                )

            total_loss  += losses["total"].item()
            total_sisdr += sisdr_i
            total_spec  += losses.get("spec_loss", torch.tensor(0.0)).item()
            n_batches   += 1

    n = max(n_batches, 1)
    return {
        "loss":    total_loss  / n,
        "si_sdri": total_sisdr / n,
        "spec":    total_spec  / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"\n[pre-train] Loading WHAM! from {cfg.wham_root} ...")

    train_loader, val_loader = build_wham_dataloaders(
        wham_root   = cfg.wham_root,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        max_train   = getattr(cfg, "max_train", None),
    )

    # GRU mode — fast pre-training, encoder/decoder weights transfer to SNN
    model = SNNTasNet(
        n_filters  = cfg.n_filters,
        kernel_sz  = cfg.kernel_sz,
        stride     = cfg.stride,
        hidden     = cfg.hidden,
        n_layers   = cfg.n_layers,
        n_speakers = 2,
        dropout    = cfg.dropout,
        snn_mode   = "gru",
        chunk_size = cfg.chunk_size,
    ).to(device)

    print(f"\n[model] Pre-training in GRU mode")
    print(f"  Params: {count_parameters(model):,}")
    print(f"  Encoder will transfer to SNN fine-tuning")
    print(f"  n_filters={cfg.n_filters}  hidden={cfg.hidden}  "
          f"layers={cfg.n_layers}  chunk={cfg.chunk_size}")
    print(f"  Epochs: {cfg.n_epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")

    loss_fn   = SeparationLoss(lambda_rate=0.0, lambda_spec=cfg.lambda_spec)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=len(train_loader),
    )
    ema    = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer     = SummaryWriter(log_dir=cfg.log_dir)
    best_val   = -1e9
    best_ckpt  = Path(cfg.checkpoint_dir) / "best_pretrain.pt"
    csv_path   = Path(cfg.csv_path)
    start_epoch = 1

    if cfg.resume and best_ckpt.exists():
        print(f"\n[resume] Loading {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "ema_state"       in ckpt: ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_si_sdri", -1e9)
        print(f"[resume] Epoch {start_epoch}  best val={best_val:+.2f} dB")

    # Init CSV
    write_header = not cfg.resume or not csv_path.exists()
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "trn_loss", "trn_si_sdri_db", "val_si_sdri_db",
                "trn_val_gap_db", "spec_loss", "lr", "epoch_time_s", "is_best",
            ])

    epoch_times: deque = deque(maxlen=10)

    print(f"\n{'─'*90}")
    print(f"{'Ep':>4}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  {'ValSI-SDRi':>10}  "
          f"{'Gap':>7}  {'Spec':>6}  {'LR':>8}  {'t':>5}  {'ETA':>8}")
    print(f"{'─'*90}")

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()
        ws = warmup_sched if epoch == 1 else None

        tr = run_epoch(model, loss_fn, train_loader, optimizer,
                       device, scaler, cfg.grad_clip, is_train=True,
                       ema=ema, warmup_sched=ws)

        ema.apply_shadow()
        vl = run_epoch(model, loss_fn, val_loader, optimizer=None,
                       device=device, scaler=None, grad_clip=0, is_train=False)
        ema.restore()

        scheduler.step(epoch)

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        gap = tr["si_sdri"] - vl["si_sdri"]

        epoch_times.append(dt)
        avg_t  = sum(epoch_times) / len(epoch_times)
        remain = avg_t * (cfg.n_epochs - epoch)
        eta    = f"{int(remain//3600)}h{int((remain%3600)//60):02d}m"

        print(f"{epoch:>4}/{cfg.n_epochs}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB  "
              f"{vl['spec']:>6.3f}  {lr:.2e}  {dt:>4.1f}s  {eta:>8}")

        writer.add_scalars("si_sdri", {"train": tr["si_sdri"], "val": vl["si_sdri"]}, epoch)
        writer.add_scalar("loss/train", tr["loss"], epoch)
        writer.add_scalar("lr", lr, epoch)

        is_best = vl["si_sdri"] > best_val
        if is_best:
            best_val = vl["si_sdri"]
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_si_sdri":     best_val,
                "model_cfg": {
                    "n_filters": cfg.n_filters, "kernel_sz": cfg.kernel_sz,
                    "stride":    cfg.stride,    "hidden":    cfg.hidden,
                    "n_layers":  cfg.n_layers,  "dropout":   cfg.dropout,
                    "snn_mode":  "gru",         "chunk_size": cfg.chunk_size,
                },
            }, best_ckpt)
            print(f"  ✔ Best pretrain checkpoint  (val={best_val:+.2f} dB)")

        if epoch % cfg.checkpoint_every == 0:
            per_ckpt = Path(cfg.checkpoint_dir) / f"pretrain_ep{epoch:04d}.pt"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "ema_state": ema.state_dict(), "val_si_sdri": vl["si_sdri"],
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, per_ckpt)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(tr["loss"], 4),
                round(tr["si_sdri"], 4), round(vl["si_sdri"], 4),
                round(gap, 4), round(vl["spec"], 4),
                lr, round(dt, 1), int(is_best),
            ])

    print(f"\n[pre-train] Done. Best val SI-SDRi = {best_val:+.2f} dB")
    print(f"[pre-train] Checkpoint: {best_ckpt}")
    print(f"\nNext step — fine-tune the SNN on your 400-scene corpus:")
    print(f"  python snn_finetune.py --pretrain_ckpt {best_ckpt} --data_root ./data")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-train SNN-TasNet on WHAM!")
    for k, v in DEFAULTS.items():
        if v is None:
            p.add_argument(f"--{k}", default=v)
        elif isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    pretrain(args)