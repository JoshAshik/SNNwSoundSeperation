"""
sep_train.py — Train SNN-TasNet, 1000 epochs with full logging.

New in this version:
  1. CSV export: every epoch appends a row to training_log.csv so you
     can plot/analyse the full training history after the run completes.
     Columns: epoch, trn_loss, trn_si_sdri, val_si_sdri, trn_spec_loss,
              val_spec_loss, trn_silence, val_silence, wform_mean, lr,
              epoch_time_s, is_best

  2. Resume from checkpoint (--resume): pass --resume to pick up exactly
     where training left off — same epoch counter, optimizer state, EMA
     weights, and LR schedule. Critical for 1000-epoch runs that may be
     interrupted.

  3. Periodic checkpoints every 50 epochs: saved as checkpoint_ep050.pt,
     checkpoint_ep100.pt etc. so you never lose more than 50 epochs of
     progress if training crashes.

  4. Estimated time remaining: printed each epoch based on rolling average
     of the last 10 epoch times.

  5. Train/val gap tracking: prints the gap each epoch so you can watch
     overfitting in real time.

  6. hidden 128→192: chunked GRU is now controlling overfitting well
     enough to safely add a little more separator capacity.

  7. Cosine T0=100, Tmult=2 for 1000 epochs: restarts at 100, 300, 700.
     Previous T0=50 was tuned for 300 epochs.
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

from sep_dataset import build_sep_dataloaders
from sep_losses  import SeparationLoss
from sep_metrics import compute_metrics, fast_si_sdri
from sep_model   import SNNTasNet, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# EMA — Exponential Moving Average of model weights
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    shadow_t = decay * shadow_{t-1} + (1 - decay) * param_t

    apply_shadow() / restore() temporarily swaps in EMA weights for eval
    then restores training weights.
    """

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
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "epoch", "trn_loss", "trn_si_sdri_db", "val_si_sdri_db",
    "trn_val_gap_db", "trn_spec_loss", "val_spec_loss",
    "trn_silence_loss", "val_silence_loss",
    "wform_mean", "lr", "epoch_time_s", "is_best",
]


def init_csv(path: Path, resume: bool):
    """Create CSV with header, or skip header if resuming."""
    if not resume or not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv(path: Path, row: dict):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data_root        = "./data",
    labels_csv       = "./labels.csv",
    data_map_json    = "./original_data_map.json",
    sample_rate      = 16_000,
    n_epochs         = 1000,          # full run
    batch_size       = 4,
    lr               = 3e-4,
    weight_decay     = 1e-5,
    grad_clip        = 3.0,
    n_filters        = 256,
    kernel_sz        = 32,
    stride           = 16,
    hidden           = 192,           # was 128 — slightly more capacity now chunking controls overfit
    n_layers         = 2,
    dropout          = 0.3,
    lambda_rate      = 0.005,
    lambda_spec      = 0.5,
    chunk_size       = 250,
    cosine_t0        = 100,           # was 50 — scaled for 1000 epoch run
    cosine_tmult     = 2,             # restarts at 100, 300, 700
    ema_decay        = 0.999,
    checkpoint_dir   = "./checkpoints_sep",
    log_dir          = "./runs_sep",
    csv_path         = "./training_log.csv",
    checkpoint_every = 50,            # save a periodic checkpoint every N epochs
    max_wall_hours   = 0.0,           # 0 = unlimited; set e.g. 5.75 for a 6h SLURM job
    device           = "cuda",
    snn_mode         = "gru",
    num_workers      = 0,
    seed             = 42,
    resume           = False,         # set True to resume from best checkpoint
)


# ─────────────────────────────────────────────────────────────────────────────
# Training / eval epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loss_fn, loader, optimizer, device, scaler,
              grad_clip, is_train, ema=None, warmup_sched=None,
              grad_noise=0.0):
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_sisdr = total_spec = total_silence = 0.0
    n_batches  = 0
    wform_mean = 0.0

    with ctx:
        for batch in loader:
            mixture = batch["mixture"].to(device, non_blocking=True)
            sources = batch["sources"].to(device, non_blocking=True)
            wform_mean += mixture.abs().mean().item()

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
                if grad_noise > 0.0:
                    with torch.no_grad():
                        for p in model.parameters():
                            p.add_(torch.randn_like(p) * grad_noise)

            with torch.no_grad():
                sisdr_i = fast_si_sdri(
                    out["separated"].float(), sources.float(), mixture.float()
                )

            total_loss    += losses["total"].item()
            total_sisdr   += sisdr_i
            total_spec    += losses.get("spec_loss",    torch.tensor(0.0)).item()
            total_silence += losses.get("silence_loss", torch.tensor(0.0)).item()
            n_batches     += 1

    n = max(n_batches, 1)
    return {
        "loss":         total_loss    / n,
        "si_sdri":      total_sisdr   / n,
        "spec_loss":    total_spec    / n,
        "silence_loss": total_silence / n,
        "wform_mean":   wform_mean    / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n[data] Building dataloaders ...")
    train_loader, val_loader, test_loader = build_sep_dataloaders(
        labels_csv    = cfg.labels_csv,
        data_map_json = cfg.data_map_json,
        data_root     = cfg.data_root,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
        seed          = cfg.seed,
    )

    snn_mode = getattr(cfg, "snn_mode", "gru")
    if snn_mode == "snn" and device.type == "cpu":
        print("[model] SNN on CPU is too slow — switching to GRU.")
        snn_mode = "gru"

    model = SNNTasNet(
        n_filters  = cfg.n_filters,
        kernel_sz  = cfg.kernel_sz,
        stride     = cfg.stride,
        hidden     = cfg.hidden,
        n_layers   = cfg.n_layers,
        n_speakers = 2,
        dropout    = cfg.dropout,
        snn_mode   = snn_mode,
        chunk_size = cfg.chunk_size,
    ).to(device)

    print(f"\n[model] SNN-TasNet  params: {count_parameters(model):,}")
    print(f"  Encoder:    {cfg.n_filters} filters, kernel={cfg.kernel_sz}, stride={cfg.stride}")
    print(f"  Separator:  {cfg.n_layers} layers, hidden={cfg.hidden}, "
          f"chunks={cfg.chunk_size} frames, bidirectional=True")
    print(f"  SpecAugment: enabled (training only)")
    print(f"  Dropout: {cfg.dropout}  |  LR: {cfg.lr}  |  Epochs: {cfg.n_epochs}")
    print(f"  Losses:     SI-SDR + {cfg.lambda_spec}×Spectral + silence penalty")
    print(f"  Scheduler:  CosineAnnealingWarmRestarts "
          f"T0={cfg.cosine_t0} Tmult={cfg.cosine_tmult} "
          f"(restarts at ~{cfg.cosine_t0}, "
          f"~{cfg.cosine_t0*(1+cfg.cosine_tmult)}, "
          f"~{cfg.cosine_t0*(1+cfg.cosine_tmult+cfg.cosine_tmult**2)})")
    print(f"  EMA decay:  {cfg.ema_decay}")
    print(f"  CSV log:    {cfg.csv_path}")

    loss_fn   = SeparationLoss(lambda_rate=cfg.lambda_rate, lambda_spec=cfg.lambda_spec)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
    )
    warmup_steps = len(train_loader)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    ema    = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer      = SummaryWriter(log_dir=cfg.log_dir)
    best_val    = -1e9
    best_ckpt   = Path(cfg.checkpoint_dir) / "best_sep.pt"
    latest_ckpt = Path(cfg.checkpoint_dir) / "latest_sep.pt"
    csv_path    = Path(cfg.csv_path)
    start_epoch = 1
    job_start   = time.time()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Prefer latest_sep.pt (most recent epoch) over best_sep.pt so we never
    # re-run epochs we already completed after a wall-time exit.
    resume_ckpt = latest_ckpt if latest_ckpt.exists() else best_ckpt
    if cfg.resume and resume_ckpt.exists():
        print(f"\n[resume] Loading checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val_si_sdri", ckpt.get("val_si_sdri", -1e9))
        print(f"[resume] Resuming from epoch {start_epoch}  "
              f"(best val SI-SDRi = {best_val:+.2f} dB)")
    else:
        if cfg.resume:
            print("[resume] No checkpoint found — starting from scratch.")

    init_csv(csv_path, resume=(cfg.resume and best_ckpt.exists()))

    # Rolling window for time-remaining estimate
    epoch_times: deque = deque(maxlen=10)

    print(f"\n{'─'*100}")
    print(f"{'Ep':>5}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  {'ValSI-SDRi':>10}  "
          f"{'Gap':>7}  {'SpecLoss':>8}  {'LR':>8}  {'t':>5}  {'ETA':>9}")
    print(f"{'─'*100}")

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()

        ws         = warmup_sched if epoch == 1 else None
        grad_noise = max(0.0, 1e-5 * (1.0 - epoch / 10.0))

        tr = run_epoch(model, loss_fn, train_loader, optimizer,
                       device, scaler, cfg.grad_clip, is_train=True,
                       ema=ema, warmup_sched=ws, grad_noise=grad_noise)

        ema.apply_shadow()
        vl = run_epoch(model, loss_fn, val_loader, optimizer=None,
                       device=device, scaler=None, grad_clip=0, is_train=False)
        ema.restore()

        scheduler.step(epoch)

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        gap = tr["si_sdri"] - vl["si_sdri"]   # train/val gap (lower = less overfit)

        epoch_times.append(dt)
        avg_t      = sum(epoch_times) / len(epoch_times)
        remaining  = avg_t * (cfg.n_epochs - epoch)
        eta_h      = int(remaining // 3600)
        eta_m      = int((remaining % 3600) // 60)
        eta_str    = f"{eta_h}h{eta_m:02d}m"

        print(f"{epoch:>5}/{cfg.n_epochs}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{vl['si_sdri']:>+9.2f}dB  "
              f"{gap:>+6.1f}dB  "
              f"{vl['spec_loss']:>8.4f}  "
              f"{lr:.2e}  {dt:>4.1f}s  {eta_str:>9}")

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalars("loss",      {"train": tr["loss"]},           epoch)
        writer.add_scalars("si_sdri",   {"train": tr["si_sdri"],
                                          "val":   vl["si_sdri"]},       epoch)
        writer.add_scalars("spec_loss", {"train": tr["spec_loss"],
                                          "val":   vl["spec_loss"]},     epoch)
        writer.add_scalar("train_val_gap",          gap,                 epoch)
        writer.add_scalar("wform_mean",             vl["wform_mean"],    epoch)
        writer.add_scalar("silence_loss/train",     tr["silence_loss"],  epoch)
        writer.add_scalar("silence_loss/val",       vl["silence_loss"],  epoch)
        writer.add_scalar("lr",                     lr,                  epoch)

        # ── Best checkpoint ───────────────────────────────────────────────────
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
                    "n_filters":  cfg.n_filters,
                    "kernel_sz":  cfg.kernel_sz,
                    "stride":     cfg.stride,
                    "hidden":     cfg.hidden,
                    "n_layers":   cfg.n_layers,
                    "dropout":    cfg.dropout,
                    "snn_mode":   snn_mode,
                    "chunk_size": cfg.chunk_size,
                },
            }, best_ckpt)
            print(f"  ✔ New best  (val SI-SDRi={best_val:+.2f} dB  gap={gap:+.1f} dB)")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        # Saved every N epochs regardless of performance — safety net against crashes.
        if epoch % cfg.checkpoint_every == 0:
            per_ckpt = Path(cfg.checkpoint_dir) / f"checkpoint_ep{epoch:04d}.pt"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_si_sdri":     vl["si_sdri"],
                "model_cfg": {
                    "n_filters":  cfg.n_filters,
                    "kernel_sz":  cfg.kernel_sz,
                    "stride":     cfg.stride,
                    "hidden":     cfg.hidden,
                    "n_layers":   cfg.n_layers,
                    "dropout":    cfg.dropout,
                    "snn_mode":   snn_mode,
                    "chunk_size": cfg.chunk_size,
                },
            }, per_ckpt)

        # ── CSV row ───────────────────────────────────────────────────────────
        append_csv(csv_path, {
            "epoch":            epoch,
            "trn_loss":         round(tr["loss"],         4),
            "trn_si_sdri_db":   round(tr["si_sdri"],      4),
            "val_si_sdri_db":   round(vl["si_sdri"],      4),
            "trn_val_gap_db":   round(gap,                4),
            "trn_spec_loss":    round(tr["spec_loss"],    4),
            "val_spec_loss":    round(vl["spec_loss"],    4),
            "trn_silence_loss": round(tr["silence_loss"], 4),
            "val_silence_loss": round(vl["silence_loss"], 4),
            "wform_mean":       round(vl["wform_mean"],   4),
            "lr":               lr,
            "epoch_time_s":     round(dt, 1),
            "is_best":          int(is_best),
        })

        # ── Latest checkpoint (every epoch, used for wall-time resume) ──────────
        torch.save({
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "ema_state":          ema.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict(),
            "val_si_sdri":        vl["si_sdri"],
            "best_val_si_sdri":   best_val,
            "model_cfg": {
                "n_filters":  cfg.n_filters,
                "kernel_sz":  cfg.kernel_sz,
                "stride":     cfg.stride,
                "hidden":     cfg.hidden,
                "n_layers":   cfg.n_layers,
                "dropout":    cfg.dropout,
                "snn_mode":   snn_mode,
                "chunk_size": cfg.chunk_size,
            },
        }, latest_ckpt)

        # ── Wall-time guard ───────────────────────────────────────────────────
        if cfg.max_wall_hours > 0:
            wall_elapsed = time.time() - job_start
            if wall_elapsed >= cfg.max_wall_hours * 3600:
                print(f"\n[wall-time] {wall_elapsed/3600:.2f}h elapsed — "
                      f"checkpoint saved to {latest_ckpt}")
                print(f"[wall-time] Resubmit with --resume to continue from epoch {epoch + 1}.")
                writer.close()
                return model

        if epoch == 3 and vl["wform_mean"] < 1e-4:
            print("\n  ⚠ WARNING: waveform_mean ≈ 0 — audio files may be missing!\n")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n[eval] Loading best checkpoint (val SI-SDRi={best_val:+.2f} dB) ...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])

    print("[eval] Computing test metrics (using EMA weights) ...")
    ema.apply_shadow()
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in test_loader:
            mix = batch["mixture"].to(device)
            src = batch["sources"].to(device)
            out = model(mix)
            m = compute_metrics(
                out["separated"].cpu(), src.cpu(), mix.cpu(),
                sample_rate=cfg.sample_rate,
            )
            all_metrics.append(m)
    ema.restore()

    keys  = set(all_metrics[0].keys())
    for m in all_metrics[1:]:
        keys &= set(m.keys())
    final = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    print(f"\n{'═'*50}")
    print(f"  TEST RESULTS  (EMA weights)")
    print(f"{'─'*50}")
    print(f"  SI-SDRi : {final['si_sdri']:+.2f} dB")
    if "sdri" in final:
        print(f"  SDRi    : {final['sdri']:+.2f} dB")
    if "pesq" in final:
        print(f"  PESQ    : {final['pesq']:.3f}  (1.0–4.5)")
    print(f"{'═'*50}")
    print(f"\n  Full epoch log saved to: {csv_path}")
    print(f"  Periodic checkpoints in: {cfg.checkpoint_dir}/")

    writer.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SNN-TasNet on 400-scene corpus")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    train(args)