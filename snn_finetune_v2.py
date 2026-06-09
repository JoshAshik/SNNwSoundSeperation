"""
snn_finetune_v2.py -- Two-phase training: GRU teacher -> SNN student with
knowledge distillation, mixup augmentation, and membrane readout.

This replaces snn_finetune.py with a training strategy that addresses the
fundamental problems in the original pipeline:

  1. Phase 1 -- GRU Teacher (50 epochs):
       Fine-tune the pre-trained GRU model on the 400-scene corpus.
       The GRU has continuous gating and hidden states, making it far more
       expressive than LIF neurons. This produces a teacher model whose
       mask outputs serve as soft targets for the SNN.

  2. Phase 2 -- SNN Student with Distillation (300 epochs):
       Train the SNN separator with a combined loss:
         total = separation_loss + lambda_distill * mask_distillation_loss
       The teacher's continuous masks guide the SNN to produce good masks
       even though its LIF neurons produce binary spikes. The membrane
       readout architecture (hidden*2 output projection) gives the SNN
       access to both spike and membrane state for mask generation.

Additional improvements:
  - Batch-level mixup augmentation (creates genuinely new examples)
  - Stronger regularisation (dropout=0.4, weight_decay=5e-4)
  - Patience-based early stopping
  - Distillation weight annealing (2.0 -> 0.5)

Usage:
    # Full pipeline (Phase 1 + Phase 2):
    python snn_finetune_v2.py --pretrain_ckpt checkpoints_pretrain\\best_pretrain.pt --data_root ./data

    # Skip Phase 1 if teacher already trained:
    python snn_finetune_v2.py --pretrain_ckpt checkpoints_pretrain\\best_pretrain.pt --data_root ./data \\
                              --teacher_ckpt checkpoints_v2\\best_gru_teacher.pt

    # Resume Phase 2 from checkpoint:
    python snn_finetune_v2.py --pretrain_ckpt checkpoints_pretrain\\best_pretrain.pt --data_root ./data \\
                              --teacher_ckpt checkpoints_v2\\best_gru_teacher.pt --resume
"""

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from sep_dataset import build_sep_dataloaders
from sep_losses  import SeparationLoss, pit_mask_distillation_loss
from sep_metrics import compute_metrics, fast_si_sdri
from sep_model   import SNNTasNet, transfer_encoder_decoder, count_parameters


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Mixup
# -----------------------------------------------------------------------------

def mixup_batch(mixture, sources, alpha=0.4):
    """
    Batch-level mixup: blend pairs of samples with random mixing coefficient.

    For each pair (i, perm[i]) in the batch, create:
        mix_new = lam * mix_i + (1-lam) * mix_perm[i]
        src_new = lam * src_i + (1-lam) * src_perm[i]

    Uses Beta(alpha, alpha) distribution for lam.
    Returns augmented (mixture, sources).
    """
    B = mixture.shape[0]
    if B < 2:
        return mixture, sources

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)  # keep lam in [0.5, 1.0] to stay close to original

    perm = torch.randperm(B, device=mixture.device)
    mixture_mixed = lam * mixture + (1.0 - lam) * mixture[perm]
    sources_mixed = lam * sources + (1.0 - lam) * sources[perm]

    # Re-normalise to prevent clipping
    # mixture is (B, T), sources is (B, C, T) -- need unsqueeze for broadcasting
    peak = mixture_mixed.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    mixture_mixed = mixture_mixed / peak
    sources_mixed = sources_mixed / peak.unsqueeze(1)

    return mixture_mixed, sources_mixed


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULTS = dict(
    pretrain_ckpt  = "./checkpoints_pretrain/best_pretrain.pt",
    teacher_ckpt   = None,            # skip Phase 1 if provided
    data_root      = "./data",
    labels_csv     = "./labels.csv",
    data_map_json  = "./original_data_map.json",
    sample_rate    = 16_000,
    # Phase 1 -- GRU teacher
    gru_epochs     = 50,
    gru_lr         = 3e-4,
    gru_freeze     = 10,              # freeze encoder/decoder for GRU warmup
    # Phase 2 -- SNN student
    snn_epochs     = 300,
    snn_lr         = 2e-4,
    snn_lr_finetune= 5e-5,           # encoder LR after unfreeze
    snn_freeze     = 30,
    # Distillation
    lambda_distill_start = 2.0,
    lambda_distill_end   = 0.5,
    # Architecture (must match pretrain checkpoint)
    n_filters      = 256,
    kernel_sz      = 32,
    stride         = 16,
    hidden         = 192,
    n_layers       = 2,
    dropout        = 0.4,            # increased from 0.3
    snn_chunk      = 100,
    # Training
    batch_size     = 4,
    weight_decay   = 5e-4,           # increased from 1e-5
    grad_clip      = 5.0,
    lambda_rate    = 0.005,
    lambda_spec    = 0.5,
    cosine_t0      = 50,
    cosine_tmult   = 2,
    ema_decay      = 0.999,
    mixup_prob     = 0.5,
    mixup_alpha    = 0.4,
    patience       = 50,
    # Paths
    checkpoint_dir = "./checkpoints_v2",
    log_dir        = "./runs_v2",
    csv_path       = "./snn_v2_log.csv",
    checkpoint_every = 25,
    device         = "cuda",
    num_workers    = 0,
    seed           = 42,
    resume         = False,
)


# -----------------------------------------------------------------------------
# Freeze / unfreeze helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Training epoch
# -----------------------------------------------------------------------------

def run_epoch(model, loss_fn, loader, optimizer, device, scaler,
              grad_clip, is_train, ema=None, mixup_prob=0.0,
              mixup_alpha=0.4, teacher=None, lambda_distill=0.0):
    """
    Run one training or validation epoch.

    If teacher is provided, computes mask distillation loss in addition
    to the standard separation loss.
    """
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_sisdr = total_spec = total_silence = total_distill = 0.0
    n_batches  = 0

    with ctx:
        for batch in loader:
            mixture = batch["mixture"].to(device, non_blocking=True)
            sources = batch["sources"].to(device, non_blocking=True)

            # Batch-level mixup (training only)
            if is_train and mixup_prob > 0 and random.random() < mixup_prob:
                mixture, sources = mixup_batch(mixture, sources, mixup_alpha)

            with autocast("cuda", enabled=(scaler is not None)):
                out    = model(mixture)
                losses = loss_fn(out["separated"], sources, out["spike_recs"])

                # Knowledge distillation from teacher
                distill_loss = torch.tensor(0.0, device=device)
                if teacher is not None and lambda_distill > 0:
                    with torch.no_grad():
                        teacher_out = teacher(mixture)
                    distill_loss = pit_mask_distillation_loss(
                        out["masks"], teacher_out["masks"].detach()
                    )
                    losses["total"] = losses["total"] + lambda_distill * distill_loss

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

            with torch.no_grad():
                sisdr_i = fast_si_sdri(
                    out["separated"].float(), sources.float(), mixture.float()
                )

            total_loss    += losses["total"].item()
            total_sisdr   += sisdr_i
            total_spec    += losses.get("spec_loss",    torch.tensor(0.0)).item()
            total_silence += losses.get("silence_loss", torch.tensor(0.0)).item()
            total_distill += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else 0.0
            n_batches     += 1

    n = max(n_batches, 1)
    return {
        "loss":         total_loss    / n,
        "si_sdri":      total_sisdr   / n,
        "spec_loss":    total_spec    / n,
        "silence_loss": total_silence / n,
        "distill_loss": total_distill / n,
    }


# -----------------------------------------------------------------------------
# Load pre-trained encoder/decoder into a model
# -----------------------------------------------------------------------------

def load_pretrained_into(model, pretrain_ckpt, device):
    """Load pre-trained encoder/decoder weights into model."""
    ckpt_path = Path(pretrain_ckpt)
    if not ckpt_path.exists():
        print(f"  [WARNING] No pre-train checkpoint at {ckpt_path}")
        return

    print(f"  [transfer] Loading pre-trained weights from {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)

    src_cfg = ckpt_data.get("model_cfg", {})
    src_model = SNNTasNet(
        n_filters=src_cfg.get("n_filters", 256),
        kernel_sz=src_cfg.get("kernel_sz", 32),
        stride=src_cfg.get("stride", 16),
        hidden=src_cfg.get("hidden", 192),
        n_layers=src_cfg.get("n_layers", 2),
        snn_mode="gru",
    ).to(device)

    if "ema_state" in ckpt_data:
        print("  [transfer] Using EMA weights from pre-training")
        for n, p in src_model.named_parameters():
            if n in ckpt_data["ema_state"]["shadow"]:
                p.data.copy_(ckpt_data["ema_state"]["shadow"][n])
    else:
        src_model.load_state_dict(ckpt_data["model_state"])

    pretrain_val = ckpt_data.get("val_si_sdri", float("nan"))
    print(f"  [transfer] Pre-trained val SI-SDRi: {pretrain_val:+.2f} dB")

    transfer_encoder_decoder(src_model, model)
    del src_model


# -----------------------------------------------------------------------------
# Phase 1: GRU Teacher
# -----------------------------------------------------------------------------

def train_gru_teacher(cfg, train_loader, val_loader, device):
    """
    Phase 1: Fine-tune a GRU model on the 400-scene corpus.

    The GRU's continuous gating produces much better separation masks than
    LIF neurons. This teacher model provides soft targets for SNN distillation.
    """
    print("\n" + "=" * 80)
    print("  PHASE 1 -- GRU TEACHER TRAINING")
    print("=" * 80)

    model = SNNTasNet(
        n_filters=cfg.n_filters, kernel_sz=cfg.kernel_sz, stride=cfg.stride,
        hidden=cfg.hidden, n_layers=cfg.n_layers, n_speakers=2,
        dropout=cfg.dropout, snn_mode="gru", chunk_size=250,
    ).to(device)

    load_pretrained_into(model, cfg.pretrain_ckpt, device)

    print(f"\n  [GRU teacher] params={count_parameters(model):,}")
    print(f"  Training for {cfg.gru_epochs} epochs with mixup")

    loss_fn = SeparationLoss(lambda_rate=0.0, lambda_spec=cfg.lambda_spec)
    ema     = EMA(model, decay=cfg.ema_decay)
    scaler  = GradScaler("cuda") if device.type == "cuda" else None

    # Freeze encoder/decoder briefly to let GRU separator warm up
    freeze_encoder_decoder(model)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.gru_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
    )

    best_val   = -1e9
    best_ckpt  = Path(cfg.checkpoint_dir) / "best_gru_teacher.pt"
    unfrozen   = False

    print(f"\n{'-'*90}")
    print(f"{'Ep':>4}  {'Phase':>7}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'LR':>8}  {'t':>5}")
    print(f"{'-'*90}")

    for epoch in range(1, cfg.gru_epochs + 1):
        t0 = time.time()

        if epoch == cfg.gru_freeze + 1 and not unfrozen:
            unfreeze_all(model)
            optimizer = Adam([
                {"params": model.separator.parameters(), "lr": cfg.gru_lr},
                {"params": list(model.encoder.parameters()) +
                           list(model.decoder.parameters()),
                 "lr": cfg.gru_lr * 0.3},
            ], weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
            )
            unfrozen = True

        phase = "freeze" if epoch <= cfg.gru_freeze else "finetune"

        tr = run_epoch(model, loss_fn, train_loader, optimizer, device, scaler,
                       cfg.grad_clip, is_train=True, ema=ema,
                       mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha)

        ema.apply_shadow()
        vl = run_epoch(model, loss_fn, val_loader, None, device, None, 0,
                       is_train=False)
        ema.restore()

        scheduler.step(epoch)
        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        gap = tr["si_sdri"] - vl["si_sdri"]

        print(f"{epoch:>4}/{cfg.gru_epochs}  {phase:>7}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB  "
              f"{lr:.2e}  {dt:>4.1f}s")

        if vl["si_sdri"] > best_val:
            best_val = vl["si_sdri"]
            torch.save({
                "epoch": epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_si_sdri":     best_val,
                "model_cfg": {
                    "n_filters": cfg.n_filters, "kernel_sz": cfg.kernel_sz,
                    "stride":    cfg.stride,    "hidden":    cfg.hidden,
                    "n_layers":  cfg.n_layers,  "dropout":   cfg.dropout,
                    "snn_mode":  "gru",         "chunk_size": 250,
                },
            }, best_ckpt)
            print(f"  -> Best GRU teacher (val={best_val:+.2f} dB)")

    print(f"\n  [Phase 1 done] Best GRU teacher: val SI-SDRi = {best_val:+.2f} dB")
    print(f"  Saved to: {best_ckpt}")
    return str(best_ckpt)


# -----------------------------------------------------------------------------
# Phase 2: SNN Student with Distillation
# -----------------------------------------------------------------------------

def train_snn_student(cfg, train_loader, val_loader, test_loader,
                      teacher_ckpt, device):
    """
    Phase 2: Train SNN with knowledge distillation from GRU teacher.

    The teacher provides soft mask targets via pit_mask_distillation_loss.
    Distillation weight anneals from lambda_distill_start -> lambda_distill_end.
    """
    print("\n" + "=" * 80)
    print("  PHASE 2 -- SNN STUDENT TRAINING (with distillation)")
    print("=" * 80)

    # -- Load teacher ----------------------------------------------------------
    teacher_path = Path(teacher_ckpt)
    print(f"\n  [teacher] Loading GRU teacher from {teacher_path}")
    teacher_data = torch.load(teacher_path, map_location=device, weights_only=False)
    teacher_cfg  = teacher_data.get("model_cfg", {})

    teacher = SNNTasNet(
        n_filters=teacher_cfg.get("n_filters", cfg.n_filters),
        kernel_sz=teacher_cfg.get("kernel_sz", cfg.kernel_sz),
        stride=teacher_cfg.get("stride", cfg.stride),
        hidden=teacher_cfg.get("hidden", cfg.hidden),
        n_layers=teacher_cfg.get("n_layers", cfg.n_layers),
        snn_mode="gru",
        chunk_size=teacher_cfg.get("chunk_size", 250),
    ).to(device)

    # Load EMA weights (best quality)
    if "ema_state" in teacher_data:
        for n, p in teacher.named_parameters():
            if n in teacher_data["ema_state"]["shadow"]:
                p.data.copy_(teacher_data["ema_state"]["shadow"][n])
    else:
        teacher.load_state_dict(teacher_data["model_state"])

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_val = teacher_data.get("val_si_sdri", float("nan"))
    print(f"  [teacher] val SI-SDRi: {teacher_val:+.2f} dB  "
          f"params={sum(p.numel() for p in teacher.parameters()):,}")

    # -- Build SNN student -----------------------------------------------------
    model = SNNTasNet(
        n_filters=cfg.n_filters, kernel_sz=cfg.kernel_sz, stride=cfg.stride,
        hidden=cfg.hidden, n_layers=cfg.n_layers, n_speakers=2,
        dropout=cfg.dropout, snn_mode="snn", snn_chunk=cfg.snn_chunk,
    ).to(device)

    load_pretrained_into(model, cfg.pretrain_ckpt, device)

    print(f"\n  [SNN student] params={count_parameters(model):,}")
    print(f"  Distillation: lambda {cfg.lambda_distill_start:.1f} -> "
          f"{cfg.lambda_distill_end:.1f}")
    print(f"  Mixup: prob={cfg.mixup_prob}  alpha={cfg.mixup_alpha}")
    print(f"  Freeze encoder/decoder for {cfg.snn_freeze} epochs")
    print(f"  Early stopping patience: {cfg.patience}")

    # -- Loss, EMA, scaler -----------------------------------------------------
    loss_fn = SeparationLoss(lambda_rate=cfg.lambda_rate, lambda_spec=cfg.lambda_spec)
    ema     = EMA(model, decay=cfg.ema_decay)
    scaler  = GradScaler("cuda") if device.type == "cuda" else None

    best_ckpt = Path(cfg.checkpoint_dir) / "best_snn.pt"
    csv_path  = Path(cfg.csv_path)
    writer    = SummaryWriter(log_dir=cfg.log_dir)
    best_val  = -1e9
    start_epoch = 1
    unfrozen  = False
    no_improve_count = 0

    # -- Build optimizer (check if resuming past freeze) -----------------------
    resume_past_freeze = False
    if cfg.resume and best_ckpt.exists():
        _peek = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        resume_past_freeze = (_peek.get("epoch", 0) + 1) > cfg.snn_freeze
        del _peek

    if resume_past_freeze:
        unfreeze_all(model)
        unfrozen = True
        optimizer = Adam([
            {"params": model.separator.parameters(), "lr": cfg.snn_lr},
            {"params": list(model.encoder.parameters()) +
                       list(model.decoder.parameters()),
             "lr": cfg.snn_lr_finetune},
        ], weight_decay=cfg.weight_decay)
    else:
        freeze_encoder_decoder(model)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.snn_lr, weight_decay=cfg.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
    )

    # -- Resume ----------------------------------------------------------------
    if cfg.resume and best_ckpt.exists():
        print(f"\n  [resume] Loading {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "ema_state"       in ckpt: ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_si_sdri", -1e9)
        print(f"  [resume] Epoch {start_epoch}  best={best_val:+.2f} dB")

    # -- CSV header ------------------------------------------------------------
    write_header = not cfg.resume or not csv_path.exists()
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "phase", "trn_loss", "trn_si_sdri_db", "val_si_sdri_db",
                "trn_val_gap_db", "distill_loss", "lambda_distill",
                "lr", "epoch_time_s", "is_best",
            ])

    epoch_times: deque = deque(maxlen=10)

    print(f"\n{'-'*110}")
    print(f"{'Ep':>5}  {'Phase':>7}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'Distill':>7}  "
          f"{'LR':>8}  {'t':>5}  {'ETA':>8}")
    print(f"{'-'*110}")

    for epoch in range(start_epoch, cfg.snn_epochs + 1):
        t0 = time.time()

        # -- Unfreeze after freeze_epochs --------------------------------------
        if epoch == cfg.snn_freeze + 1 and not unfrozen:
            print(f"\n  [epoch {epoch}] Unfreezing encoder+decoder")
            unfreeze_all(model)
            optimizer = Adam([
                {"params": model.separator.parameters(), "lr": cfg.snn_lr},
                {"params": list(model.encoder.parameters()) +
                           list(model.decoder.parameters()),
                 "lr": cfg.snn_lr_finetune},
            ], weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=1e-6,
            )
            unfrozen = True

        phase = "freeze" if epoch <= cfg.snn_freeze else "finetune"

        # -- Anneal distillation weight ----------------------------------------
        progress = min(1.0, (epoch - 1) / max(1, cfg.snn_epochs - 1))
        lambda_d = (cfg.lambda_distill_start
                    + progress * (cfg.lambda_distill_end - cfg.lambda_distill_start))

        # -- Train -------------------------------------------------------------
        tr = run_epoch(model, loss_fn, train_loader, optimizer, device, scaler,
                       cfg.grad_clip, is_train=True, ema=ema,
                       mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha,
                       teacher=teacher, lambda_distill=lambda_d)

        # -- Validate (with EMA weights) ---------------------------------------
        ema.apply_shadow()
        vl = run_epoch(model, loss_fn, val_loader, None, device, None, 0,
                       is_train=False,
                       teacher=teacher, lambda_distill=lambda_d)
        ema.restore()

        scheduler.step(epoch)

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        gap = tr["si_sdri"] - vl["si_sdri"]

        epoch_times.append(dt)
        avg_t  = sum(epoch_times) / len(epoch_times)
        remain = avg_t * (cfg.snn_epochs - epoch)
        eta    = f"{int(remain//3600)}h{int((remain%3600)//60):02d}m"

        print(f"{epoch:>5}/{cfg.snn_epochs}  {phase:>7}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB  "
              f"{vl['distill_loss']:>7.4f}  "
              f"{lr:.2e}  {dt:>4.1f}s  {eta:>8}")

        writer.add_scalars("si_sdri", {"train": tr["si_sdri"], "val": vl["si_sdri"]}, epoch)
        writer.add_scalar("train_val_gap", gap, epoch)
        writer.add_scalar("distill_loss/val", vl["distill_loss"], epoch)
        writer.add_scalar("lambda_distill", lambda_d, epoch)
        writer.add_scalar("lr", lr, epoch)

        is_best = vl["si_sdri"] > best_val
        if is_best:
            best_val = vl["si_sdri"]
            no_improve_count = 0
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
                    "snn_mode":  "snn",         "snn_chunk": cfg.snn_chunk,
                },
            }, best_ckpt)
            print(f"  -> Best SNN (val={best_val:+.2f} dB  gap={gap:+.1f} dB)")
        else:
            no_improve_count += 1

        if epoch % cfg.checkpoint_every == 0:
            per = Path(cfg.checkpoint_dir) / f"snn_ep{epoch:04d}.pt"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "ema_state": ema.state_dict(), "val_si_sdri": vl["si_sdri"],
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, per)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase,
                round(tr["loss"], 4), round(tr["si_sdri"], 4), round(vl["si_sdri"], 4),
                round(gap, 4), round(vl["distill_loss"], 4), round(lambda_d, 4),
                lr, round(dt, 1), int(is_best),
            ])

        # -- Early stopping ----------------------------------------------------
        if no_improve_count >= cfg.patience and epoch > cfg.snn_freeze + 50:
            print(f"\n  [early stop] No improvement for {cfg.patience} epochs. Stopping.")
            break

    # -- Test evaluation -------------------------------------------------------
    print(f"\n[eval] Best checkpoint: val SI-SDRi = {best_val:+.2f} dB")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])

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

    print(f"\n{'='*50}")
    print(f"  TEST RESULTS  (SNN + EMA weights)")
    print(f"{'-'*50}")
    print(f"  SI-SDRi : {final['si_sdri']:+.2f} dB")
    if "sdri" in final:
        print(f"  SDRi    : {final['sdri']:+.2f} dB")
    if "pesq" in final:
        print(f"  PESQ    : {final['pesq']:.3f}")
    print(f"{'='*50}")
    print(f"\n  Log: {csv_path}")
    print(f"  Checkpoints: {cfg.checkpoint_dir}/")
    print(f"\n  Run inference:")
    print(f"    python sep_inference.py --checkpoint {best_ckpt} "
          f"--from_dataset --data_root {cfg.data_root}")

    writer.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- Data ------------------------------------------------------------------
    print("\n[data] Building 400-scene dataloaders ...")
    train_loader, val_loader, test_loader = build_sep_dataloaders(
        labels_csv    = cfg.labels_csv,
        data_map_json = cfg.data_map_json,
        data_root     = cfg.data_root,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
        seed          = cfg.seed,
    )

    # -- Phase 1: GRU Teacher -------------------------------------------------
    teacher_ckpt = cfg.teacher_ckpt
    if teacher_ckpt is None or not Path(teacher_ckpt).exists():
        teacher_ckpt = train_gru_teacher(cfg, train_loader, val_loader, device)
    else:
        print(f"\n[skip Phase 1] Using existing teacher: {teacher_ckpt}")

    # -- Phase 2: SNN Student -------------------------------------------------
    train_snn_student(cfg, train_loader, val_loader, test_loader,
                      teacher_ckpt, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SNN fine-tuning v2: GRU teacher -> SNN student with distillation"
    )
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    main(args)
