"""
snn_finetune.py — Fine-tune SNN-TasNet on 400-scene corpus after WHAM! pre-training.

This is the true spiking neural network training. It:
  1. Loads the GRU pre-trained checkpoint (encoder + decoder already trained)
  2. Builds a new model in SNN mode with ChunkedSNNSeparator
  3. Transfers encoder + decoder weights from pre-training
  4. Freezes encoder + decoder for freeze_epochs (default 20)
     → Only the SNN separator trains at first, learning to use the
       pre-trained encoder features without destroying them
  5. Unfreezes everything and trains jointly for remaining epochs

Why freeze first:
  When you transfer pre-trained encoder/decoder weights and immediately train
  everything with a random SNN separator, the large gradients from the
  untrained separator flow back through the encoder and corrupt the pre-trained
  features. Freezing for 20 epochs lets the SNN separator warm up to a
  reasonable starting point before the encoder gets touched.

After freeze_epochs:
  All parameters unfreeze and train jointly at a lower LR (lr_finetune).
  The encoder adapts to the different sound classes in the 400-scene corpus
  while the SNN separator refines its masking.

Expected outcomes vs previous 300-scene-only training:
  Previous:   val SI-SDRi ≈ -14.9 dB
  After WHAM! pre-training + SNN finetune:  expected -5 to 0 dB
  (this is a rough estimate; actual improvement depends on WHAM! convergence)

Usage:
    python snn_finetune.py --pretrain_ckpt ./checkpoints_pretrain/best_pretrain.pt
                           --data_root ./data

The fine-tuned checkpoint is saved to ./checkpoints_snn/best_snn.pt and is
compatible with sep_inference.py (reads model_cfg automatically).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress TF/oneDNN noise before any import

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

# Windows console may default to cp1252 — force UTF-8 so box-drawing chars work
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from sep_dataset   import build_sep_dataloaders
from fsd50k_dataset import build_fsd50k_dataloaders
from sep_losses  import SeparationLoss
from sep_metrics import compute_metrics, fast_si_sdri, diagnostic_metrics, print_diagnostics
from sep_model   import SNNTasNet, transfer_encoder_decoder, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model   = model
        self.decay   = decay
        self.shadow  = {n: p.data.clone() for n, p in model.named_parameters()}
        self._backup = {}

    def update(self):
        # lerp_ is in-place: shadow = shadow + (1-decay)*(p - shadow) = decay*shadow + (1-decay)*p
        # Avoids allocating a new tensor per parameter per batch.
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.shadow[n].lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self):
        # Swap data pointers — no cloning of model params (saves ~5 MB per epoch).
        # During eval (torch.no_grad, model.eval) nothing writes to p.data, so
        # pointing it at shadow is safe. restore() swaps back the original tensor.
        self._backup = {}
        for n, p in self.model.named_parameters():
            self._backup[n] = p.data          # store original tensor pointer
            p.data = self.shadow[n]           # point model weights at shadow

    def restore(self):
        for n, p in self.model.named_parameters():
            p.data = self._backup[n]          # restore original tensor pointer

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, sd):
        self.shadow = sd["shadow"]
        self.decay  = sd["decay"]


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    pretrain_ckpt  = "./checkpoints_pretrain/best_pretrain.pt",
    # ── dataset selection ────────────────────────────────────────────────────
    dataset        = "fsd50k",       # "fsd50k" (v2) or "original" (400-scene corpus)
    fsd50k_root    = "./data/fsd50k",
    scenes_per_epoch = 10_000,       # fsd50k only: synthetic scenes generated per epoch
    # ── original dataset paths (used only when dataset="original") ───────────
    data_root      = "./data",
    labels_csv     = "./labels.csv",
    data_map_json  = "./original_data_map.json",
    # ── training ─────────────────────────────────────────────────────────────
    sample_rate    = 16_000,
    n_epochs       = 1000,
    freeze_epochs  = 20,             # freeze encoder/decoder while SNN separator warms up
    batch_size     = 4,
    lr             = 1.5e-4,         # v2: halved from 3e-4 — lower LR for stability
    lr_finetune    = 5e-5,           # v2: halved from 1e-4 — maintain 1:3 ratio with lr
    weight_decay   = 1e-5,
    grad_clip      = 3.0,
    # ── model dims — fresh start; resume reads arch from checkpoint ──────────
    n_filters      = 256,
    kernel_sz      = 32,
    stride         = 16,
    hidden         = 256,
    n_layers       = 3,
    n_speakers     = 8,
    dropout        = 0.3,
    snn_chunk      = 100,
    # ── loss weights ─────────────────────────────────────────────────────────
    lambda_rate    = 0.03,           # v2: raised from 0.005 — combat spike saturation
    lambda_spec    = 0.5,
    lambda_recon   = 0.5,            # v2: halved from 1.0 — v1 over-corrected energy
    # ── infrastructure ───────────────────────────────────────────────────────
    ema_decay      = 0.999,
    checkpoint_dir = "./checkpoints_snn",
    log_dir        = "./runs_snn",
    csv_path       = "./snn_finetune_log.csv",   # overridden for fsd50k in code
    checkpoint_every = 50,
    val_every      = 1,             # run validation every N epochs (e.g. 5 = skip 4/5 val runs)
    max_wall_hours = 0.0,           # 0 = unlimited; set e.g. 5.75 for a 6h SLURM job
    device         = "cuda",
    snn_mode       = "snn",
    num_workers    = 0,
    seed           = 42,
    resume         = False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loss_fn, loader, optimizer, device, scaler,
              grad_clip, is_train, ema=None, warmup_sched=None,
              grad_noise=0.0):
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_sisdr = total_spec = total_silence = total_recon = 0.0
    total_mask_contrast = total_spike_rate = 0.0
    n_batches  = 0
    wform_mean = 0.0
    _t0_batch  = time.time()

    with ctx:
        for batch in loader:
            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)

            with autocast("cuda", enabled=(scaler is not None)):
                out    = model(mixture)
                losses = loss_fn(out["separated"], sources, out["spike_recs"],
                                 mixture=mixture)
                # Accumulate wform_mean inside autocast to avoid extra sync
                wform_mean += mixture.abs().mean().item()

            if is_train:
                optimizer.zero_grad(set_to_none=True)
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
                            if p.requires_grad:
                                p.add_(torch.randn_like(p) * grad_noise)

            with torch.no_grad():
                sisdr_i = fast_si_sdri(
                    out["separated"].float(), sources.float(), mixture.float()
                )

                # --- Diagnostic stats (val only to keep train fast) ---
                if not is_train:
                    masks = out.get("masks")
                    if masks is not None:
                        m = masks.float()
                        total_mask_contrast += m.std(dim=1).mean().item()

                    spike_recs = out.get("spike_recs")
                    if spike_recs:
                        total_spike_rate += sum(
                            r.float().mean().item() for r in spike_recs
                        ) / len(spike_recs)

            total_loss    += losses["total"].item()
            total_sisdr   += sisdr_i
            total_spec    += losses.get("spec_loss",    torch.tensor(0.0)).item()
            total_silence += losses.get("silence_loss", torch.tensor(0.0)).item()
            total_recon   += losses.get("recon_loss",   torch.tensor(0.0)).item()
            n_batches     += 1

            # Print per-batch timing every 100 batches so we can see training is alive
            if is_train and n_batches % 100 == 0:
                elapsed = time.time() - _t0_batch
                print(f"  [batch {n_batches}] {elapsed/100:.2f}s/batch  "
                      f"loss={losses['total'].item():.3f}", flush=True)
                _t0_batch = time.time()

    n = max(n_batches, 1)
    return {
        "loss":          total_loss    / n,
        "si_sdri":       total_sisdr   / n,
        "spec_loss":     total_spec    / n,
        "silence_loss":  total_silence / n,
        "recon_loss":    total_recon   / n,
        "wform_mean":    wform_mean    / n,
        "mask_contrast": total_mask_contrast / n,
        "spike_rate":    total_spike_rate    / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder_decoder(model: SNNTasNet):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    n_frozen = (sum(p.numel() for p in model.encoder.parameters()) +
                sum(p.numel() for p in model.decoder.parameters()))
    print(f"  [freeze] Encoder+decoder frozen ({n_frozen:,} params)")


def unfreeze_all(model: SNNTasNet):
    for p in model.parameters():
        p.requires_grad = True
    print(f"  [unfreeze] All params trainable ({count_parameters(model):,} params)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def finetune(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if cfg.dataset == "fsd50k":
        print("\n[data] Building FSD50K dataloaders ...")
        train_loader, val_loader, test_loader = build_fsd50k_dataloaders(
            fsd50k_root      = cfg.fsd50k_root,
            batch_size       = cfg.batch_size,
            num_workers      = cfg.num_workers,
            seed             = cfg.seed,
            scenes_per_epoch = cfg.scenes_per_epoch,
        )
    else:
        print("\n[data] Building 400-scene dataloaders ...")
        train_loader, val_loader, test_loader = build_sep_dataloaders(
            labels_csv    = cfg.labels_csv,
            data_map_json = cfg.data_map_json,
            data_root     = cfg.data_root,
            batch_size    = cfg.batch_size,
            num_workers   = cfg.num_workers,
            seed          = cfg.seed,
        )

    # ── Early resume: read architecture from checkpoint BEFORE building model ──
    # Must happen before SNNTasNet() so cfg.hidden/n_layers match the saved weights.
    # v2 (fsd50k) saves to best_snn_v2.pt so v1 (best_snn.pt) is never overwritten.
    _ckpt_name  = "best_snn_v2.pt" if cfg.dataset == "fsd50k" else "best_snn.pt"
    best_ckpt   = Path(cfg.checkpoint_dir) / _ckpt_name
    latest_ckpt = Path(cfg.checkpoint_dir) / _ckpt_name.replace(".pt", "_latest.pt")
    # Prefer latest (most recent epoch) over best for architecture peek
    _peek_ckpt  = latest_ckpt if latest_ckpt.exists() else best_ckpt
    if cfg.resume and _peek_ckpt.exists():
        _early = torch.load(_peek_ckpt, map_location="cpu", weights_only=False)
        _saved_cfg = _early.get("model_cfg", {})
        for _key in ("n_filters", "hidden", "n_layers", "dropout", "snn_chunk", "n_speakers"):
            if _key in _saved_cfg:
                setattr(cfg, _key, _saved_cfg[_key])
        if _saved_cfg:
            print(f"[resume] Architecture from checkpoint: "
                  f"hidden={cfg.hidden}  n_layers={cfg.n_layers}  n_speakers={cfg.n_speakers}")
        del _early

    # ── Build SNN model ───────────────────────────────────────────────────────
    snn_mode = cfg.snn_mode
    if snn_mode == "snn" and device.type == "cpu":
        print("[model] SNN on CPU is too slow — use GPU or pass --snn_mode gru")

    model = SNNTasNet(
        n_filters  = cfg.n_filters,
        kernel_sz  = cfg.kernel_sz,
        stride     = cfg.stride,
        hidden     = cfg.hidden,
        n_layers   = cfg.n_layers,
        n_speakers = cfg.n_speakers,
        dropout    = cfg.dropout,
        snn_mode   = snn_mode,
        snn_chunk  = cfg.snn_chunk,
    ).to(device)

    # ── Transfer pre-trained encoder/decoder ──────────────────────────────────
    pretrain_ckpt = Path(cfg.pretrain_ckpt)
    if pretrain_ckpt.exists():
        print(f"\n[transfer] Loading pre-trained weights from {pretrain_ckpt}")
        ckpt_data = torch.load(pretrain_ckpt, map_location=device, weights_only=False)

        # Build a temp GRU model to load the checkpoint into
        src_cfg  = ckpt_data.get("model_cfg", {})
        src_model = SNNTasNet(
            n_filters = src_cfg.get("n_filters", cfg.n_filters),
            kernel_sz = src_cfg.get("kernel_sz", cfg.kernel_sz),
            stride    = src_cfg.get("stride",    cfg.stride),
            hidden    = src_cfg.get("hidden",    cfg.hidden),
            n_layers  = src_cfg.get("n_layers",  cfg.n_layers),
            snn_mode  = "gru",
        ).to(device)

        # Load EMA weights if available (better quality than raw weights)
        if "ema_state" in ckpt_data:
            print("[transfer] Using EMA weights from pre-training")
            for n, p in src_model.named_parameters():
                if n in ckpt_data["ema_state"]["shadow"]:
                    p.data.copy_(ckpt_data["ema_state"]["shadow"][n])
        else:
            src_model.load_state_dict(ckpt_data["model_state"])

        pretrain_val = ckpt_data.get("val_si_sdri", float("nan"))
        print(f"[transfer] Pre-trained val SI-SDRi: {pretrain_val:+.2f} dB")

        transfer_encoder_decoder(src_model, model)
        del src_model

    else:
        print(f"\n[WARNING] No pre-train checkpoint at {pretrain_ckpt}")
        print(f"  Training SNN from scratch (expect slower convergence)")
        print(f"  Run wham_pretrain.py first for best results\n")

    print(f"\n[model] SNN-TasNet (snn_mode={snn_mode})  "
          f"params={count_parameters(model):,}")
    print(f"  Encoder:  {cfg.n_filters} filters  (transferred from WHAM! pre-training)")
    print(f"  Separator: ChunkedSNN  hidden={cfg.hidden}  "
          f"layers={cfg.n_layers}  chunk={cfg.snn_chunk}")
    print(f"  Freeze encoder/decoder for first {cfg.freeze_epochs} epochs")

    # ── Loss and EMA ─────────────────────────────────────────────────────────
    loss_fn = SeparationLoss(lambda_rate=cfg.lambda_rate, lambda_spec=cfg.lambda_spec,
                             lambda_recon=cfg.lambda_recon, use_pit=False)
    ema    = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer      = SummaryWriter(log_dir=cfg.log_dir)
    best_val    = -1e9
    job_start   = time.time()
    # best_ckpt / latest_ckpt already set in early resume block above
    # v2 (fsd50k) gets its own CSV so v1 log is never overwritten
    _csv_name = "snn_finetune_v2_log.csv" if cfg.dataset == "fsd50k" else cfg.csv_path
    csv_path  = Path(_csv_name)
    start_epoch = 1
    unfrozen    = False

    # ── Peek at checkpoint epoch BEFORE building optimizer ────────────────────
    # The optimizer has a different number of parameter groups depending on
    # whether we are in the freeze phase (1 group) or finetune phase (2 groups).
    # We must build the matching optimizer FIRST, then load its state dict.
    resume_past_freeze = False
    _resume_ckpt = latest_ckpt if latest_ckpt.exists() else best_ckpt
    if cfg.resume and _resume_ckpt.exists():
        _peek = torch.load(_resume_ckpt, map_location="cpu", weights_only=False)
        _saved_epoch = _peek.get("epoch", 0)
        resume_past_freeze = (_saved_epoch + 1) > cfg.freeze_epochs
        del _peek

    if resume_past_freeze:
        # Checkpoint was saved in finetune phase — need 2-group optimizer
        unfreeze_all(model)
        unfrozen = True
        optimizer = Adam([
            {"params": model.separator.parameters(), "lr": cfg.lr},
            {"params": list(model.encoder.parameters()) +
                       list(model.decoder.parameters()),
             "lr": cfg.lr_finetune},
        ], weight_decay=cfg.weight_decay)
    else:
        # Checkpoint was saved during freeze phase (or fresh start) — 1-group optimizer
        freeze_encoder_decoder(model)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    # v2: single smooth cosine decay — no warm restarts that destroy progress
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.n_epochs - cfg.freeze_epochs), eta_min=1e-6,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=len(train_loader),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    # Prefer latest_ckpt (most recent epoch) so we never re-run completed epochs.
    if cfg.resume and _resume_ckpt.exists():
        print(f"\n[resume] Loading {_resume_ckpt}")
        ckpt = torch.load(_resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "ema_state"       in ckpt: ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val_si_sdri", ckpt.get("val_si_sdri", -1e9))
        print(f"[resume] Epoch {start_epoch}  best={best_val:+.2f} dB  "
              f"phase={'finetune' if unfrozen else 'freeze'}")


    # ── CSV ───────────────────────────────────────────────────────────────────
    _header = [
        "epoch", "phase", "trn_loss", "trn_si_sdri_db", "val_si_sdri_db",
        "trn_val_gap_db", "spec_loss", "silence_loss", "recon_loss",
        "mask_contrast", "spike_rate", "wform_mean", "lr", "epoch_time_s", "is_best",
    ]
    if not cfg.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_header)
    else:
        # Ensure header is up-to-date even on resume
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
        if first_line != ",".join(_header):
            with open(csv_path, "r") as f:
                old_rows = f.readlines()[1:]
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_header)
                f.writelines(old_rows)

    epoch_times: deque = deque(maxlen=10)
    vl: dict = {}   # last validation metrics; empty until first val epoch

    print(f"\n{'─'*115}")
    print(f"{'Ep':>5}  {'Phase':>7}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'MaskCtrs':>8}  {'SpikeRate':>9}  "
          f"{'LR':>8}  {'t':>5}  {'ETA':>8}")
    print(f"{'─'*115}")

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()

        # ── Unfreeze after freeze_epochs ──────────────────────────────────────
        if epoch == cfg.freeze_epochs + 1 and not unfrozen:
            print(f"\n  [epoch {epoch}] Unfreezing encoder+decoder, "
                  f"switching to lr={cfg.lr_finetune:.1e}")
            unfreeze_all(model)
            # Rebuild optimizer so encoder/decoder are included at lower LR
            optimizer = Adam([
                {"params": model.separator.parameters(), "lr": cfg.lr},
                {"params": list(model.encoder.parameters()) +
                           list(model.decoder.parameters()),
                 "lr": cfg.lr_finetune},
            ], weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg.n_epochs - epoch), eta_min=1e-6,
            )
            unfrozen = True

        phase = "freeze" if epoch <= cfg.freeze_epochs else "finetune"
        ws    = warmup_sched if epoch == 1 else None
        gnoise = max(0.0, 1e-5 * (1.0 - epoch / 10.0))

        tr = run_epoch(model, loss_fn, train_loader, optimizer,
                       device, scaler, cfg.grad_clip, is_train=True,
                       ema=ema, warmup_sched=ws, grad_noise=gnoise)

        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.n_epochs)
        if do_val:
            ema.apply_shadow()
            vl = run_epoch(model, loss_fn, val_loader, optimizer=None,
                           device=device, scaler=None, grad_clip=0, is_train=False)
            ema.restore()
        # else: reuse last vl (initialised to None before loop; only best/CSV updated on do_val)

        scheduler.step()

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        gap = tr["si_sdri"] - (vl["si_sdri"] if do_val else float("nan"))

        epoch_times.append(dt)
        avg_t  = sum(epoch_times) / len(epoch_times)
        remain = avg_t * (cfg.n_epochs - epoch)
        eta    = f"{int(remain//3600)}h{int((remain%3600)//60):02d}m"

        if do_val:
            mc = vl["mask_contrast"]
            sr = vl["spike_rate"]
            val_str = f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB  {mc:>8.4f}  {sr:>9.4f}"
        else:
            val_str = f"{'---':>9}    {'---':>6}   {'---':>8}   {'---':>9}"
        print(f"{epoch:>5}/{cfg.n_epochs}  {phase:>7}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{val_str}  {lr:.2e}  {dt:>4.1f}s  {eta:>8}")

        writer.add_scalar("si_sdri/train", tr["si_sdri"], epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("silence_loss/train", tr["silence_loss"], epoch)
        writer.add_scalar("recon_loss/train",   tr["recon_loss"],   epoch)
        if do_val:
            writer.add_scalar("si_sdri/val",        vl["si_sdri"],       epoch)
            writer.add_scalar("train_val_gap",       gap,                 epoch)
            writer.add_scalar("recon_loss/val",      vl["recon_loss"],    epoch)
            writer.add_scalar("mask_contrast/val",   vl["mask_contrast"], epoch)
            writer.add_scalar("spike_rate/val",      vl["spike_rate"],    epoch)

        is_best = do_val and vl["si_sdri"] > best_val
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
                    "n_filters":  cfg.n_filters,  "kernel_sz": cfg.kernel_sz,
                    "stride":     cfg.stride,      "hidden":    cfg.hidden,
                    "n_layers":   cfg.n_layers,    "dropout":   cfg.dropout,
                    "snn_mode":   snn_mode,        "snn_chunk": cfg.snn_chunk,
                    "n_speakers": cfg.n_speakers,
                },
            }, best_ckpt)
            print(f"  ✔ Best SNN checkpoint  (val={best_val:+.2f} dB  gap={gap:+.1f} dB)")

        if epoch % cfg.checkpoint_every == 0:
            per = Path(cfg.checkpoint_dir) / f"snn_ep{epoch:04d}.pt"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "val_si_sdri": vl.get("si_sdri", float("nan")),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, per)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase,
                round(tr["loss"], 4), round(tr["si_sdri"], 4),
                round(vl["si_sdri"], 4) if do_val else "",
                round(gap, 4) if do_val else "",
                round(vl["spec_loss"], 4) if do_val else "",
                round(vl["silence_loss"], 4) if do_val else "",
                round(vl["recon_loss"], 4) if do_val else "",
                round(vl["mask_contrast"], 4) if do_val else "",
                round(vl["spike_rate"], 4) if do_val else "",
                round(vl["wform_mean"], 4) if do_val else "",
                lr, round(dt, 1), int(is_best),
            ])

        # ── Latest checkpoint (every epoch — used for wall-time resume) ──────────
        torch.save({
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "ema_state":        ema.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "val_si_sdri":      vl.get("si_sdri", float("nan")),
            "best_val_si_sdri": best_val,
            "model_cfg": {
                "n_filters":  cfg.n_filters,  "kernel_sz": cfg.kernel_sz,
                "stride":     cfg.stride,      "hidden":    cfg.hidden,
                "n_layers":   cfg.n_layers,    "dropout":   cfg.dropout,
                "snn_mode":   snn_mode,        "snn_chunk": cfg.snn_chunk,
                "n_speakers": cfg.n_speakers,
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

        if epoch == 3 and vl.get("wform_mean", float("nan")) < 1e-4:
            print("\n  ⚠ WARNING: waveform_mean ≈ 0 — check data_root path!\n")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n[eval] Best checkpoint: val SI-SDRi = {best_val:+.2f} dB")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])

    ema.apply_shadow()
    model.eval()
    all_metrics = []
    all_diag    = []
    with torch.no_grad():
        for batch in test_loader:
            mix = batch["mixture"].to(device)
            src = batch["sources"].to(device)
            out = model(mix)
            sep_cpu = out["separated"].cpu()
            src_cpu = src.cpu()
            mix_cpu = mix.cpu()

            m = compute_metrics(sep_cpu, src_cpu, mix_cpu,
                                sample_rate=cfg.sample_rate,
                                use_pesq=(cfg.dataset != "fsd50k"))
            all_metrics.append(m)

            d = diagnostic_metrics(
                sep_cpu, src_cpu, mix_cpu,
                masks      = out.get("masks", None),
                spike_recs = out.get("spike_recs", None),
            )
            all_diag.append(d)
    ema.restore()

    keys  = set(all_metrics[0].keys())
    for m in all_metrics[1:]:
        keys &= set(m.keys())
    final = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    # Average diagnostics across all test batches
    diag_keys = set(all_diag[0].keys())
    for d in all_diag[1:]:
        diag_keys &= set(d.keys())
    final_diag = {k: sum(d[k] for d in all_diag) / len(all_diag) for k in diag_keys}

    print(f"\n{'═'*50}")
    print(f"  TEST RESULTS  (SNN + EMA weights)")
    print(f"{'─'*50}")
    print(f"  SI-SDRi : {final['si_sdri']:+.2f} dB")
    if "sdri" in final:
        print(f"  SDRi    : {final['sdri']:+.2f} dB")
    if "pesq" in final:
        print(f"  PESQ    : {final['pesq']:.3f}")
    print(f"{'═'*50}")

    print_diagnostics(final_diag)
    print(f"\n  Log: {csv_path}")
    print(f"  Checkpoints: {cfg.checkpoint_dir}/")
    print(f"\n  Run inference:")
    print(f"    python sep_inference.py --checkpoint {best_ckpt} "
          f"--from_dataset --data_root {cfg.data_root}")

    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SNN fine-tuning after WHAM! pre-training")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    finetune(args)