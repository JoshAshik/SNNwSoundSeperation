"""
train.py â€“ Training loop for the SNN sound source separation model.

Usage:
    python train.py [--data_root PATH] [--epochs N] [--batch_size N] [--device cuda|cpu]

The script:
  1. Builds dataloaders from the provided labels.csv + original_data_map.json
  2. Instantiates the AudioToSpikes pipeline and SNNSoundSeparator model
  3. Runs training with mixed-precision (AMP), gradient clipping, and cosine LR schedule
  4. Logs metrics to TensorBoard and saves the best checkpoint
  5. Evaluates on the test set at the end

If audio files are not yet available the dataset returns silent tensors, which
lets you verify the training loop and model shapes without the full dataset.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from config import (
    snn_cfg, cochlea_cfg, audio_cfg, enc_cfg, train_cfg, TrainConfig
)
from dataset import build_dataloaders
from encoding import AudioToSpikes
from model import SNNSoundSeparator, SeparationLoss, count_parameters


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    per_class_thresholds: List[float] | None = None,
) -> Dict[str, float]:
    """Multi-label accuracy, precision, recall, F1 at a given threshold."""
    probs = logits.sigmoid()
    if per_class_thresholds is not None:
        t_tensor = torch.tensor(per_class_thresholds, device=probs.device).unsqueeze(0)
        preds = (probs >= t_tensor).float()
    else:
        preds = (probs >= threshold).float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    # Exact-match accuracy (all labels correct per sample)
    exact = (preds == labels).all(dim=1).float().mean().item()

    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact}


def find_optimal_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    steps:  int = 100,
) -> List[float]:
    """
    Search per-class for the threshold that maximises per-class F1.
    Prints sigmoid statistics to diagnose model confidence calibration.
    """
    probs = logits.sigmoid()
    n_classes = labels.shape[1]
    thresholds = []

    # Diagnostic: print sigmoid output distribution
    print(f"  [sigmoid stats]  min={probs.min():.3f}  max={probs.max():.3f}  "
          f"mean={probs.mean():.3f}  median={probs.median():.3f}")
    print(f"  [per-class sigmoid means]: "
          f"{[f'{probs[:,c].mean().item():.2f}' for c in range(n_classes)]}")

    print("  [threshold search per class]")
    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        # Search full range
        for t in torch.linspace(0.01, 0.99, steps):
            preds = (probs[:, c] >= t.item()).float()
            tp = (preds * labels[:, c]).sum()
            fp = (preds * (1 - labels[:, c])).sum()
            fn = ((1 - preds) * labels[:, c]).sum()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, t.item()
        thresholds.append(best_t)
        mean_prob = probs[:, c].mean().item()
        print(f"    cls{c}: threshold={best_t:.2f}  f1={best_f1:.3f}  "
              f"mean_sigmoid={mean_prob:.3f}")
    return thresholds


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# One epoch
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_epoch(
    model:       SNNSoundSeparator,
    encoder:     AudioToSpikes,
    loss_fn:     SeparationLoss,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer | None,
    device:      torch.device,
    scaler:      GradScaler | None,
    is_train:    bool = True,
) -> Dict[str, float]:

    model.train(is_train)
    encoder.train(is_train)

    total_losses = {"total": 0., "ce": 0., "sep": 0., "rate": 0.}
    all_logits, all_labels = [], []
    t0 = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    first_batch_printed = False
    with ctx:
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)  # (B, 2, T)
            labels   = batch["label"].to(device, non_blocking=True)     # (B, 8)

            with autocast(enabled=(scaler is not None)):
                spikes = encoder(waveform)                              # (T, B, D)
                # Print input stats once per eval pass to verify encoding health
                if not is_train and not first_batch_printed:
                    first_batch_printed = True
                    wmean = waveform.abs().mean().item()
                    smean = spikes.mean().item()
                    smax  = spikes.max().item()
                    print(f"  [encoding] waveform_mean={wmean:.4f}  "
                          f"spike_mean={smean:.4f}  spike_max={smax:.4f}")

                # Mixup augmentation during training (alpha=0.2)
                # Blends pairs of samples in spike + label space.
                # Very effective for small datasets -- creates soft labels
                # that prevent overconfident predictions.
                if is_train and spikes.shape[1] > 1:
                    alpha = 0.2
                    lam   = float(torch.distributions.Beta(alpha, alpha).sample())
                    lam   = max(lam, 1 - lam)   # ensure dominant sample > 0.5
                    idx   = torch.randperm(spikes.shape[1], device=spikes.device)
                    spikes = lam * spikes + (1 - lam) * spikes[:, idx]
                    labels = lam * labels + (1 - lam) * labels[idx]

                out    = model(spikes)
                losses = loss_fn(
                    out["logits"], labels, out["slot_embeddings"], out["spike_recs"]
                )

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(encoder.parameters()), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(encoder.parameters()), max_norm=1.0
                    )
                    optimizer.step()

            for k in total_losses:
                total_losses[k] += losses[k].item()

            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(labels.detach().cpu())

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

# ... inside run_epoch() ...

    # â”€â”€â”€ ADD THIS DEBUG BLOCK â”€â”€â”€
    if not is_train:
        preds = (all_logits.sigmoid() > 0.5).float() # Assuming 0.5 threshold
        print("\n--- DEBUG: Class Prediction Counts ---")
        print(f"Total Samples: {preds.shape[0]}")
        print(f"Predicted Counts per Class: {preds.sum(dim=0).tolist()}")
        print(f"Actual Counts per Class:    {all_labels.sum(dim=0).tolist()}")
        print("--------------------------------------\n")
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    metrics = compute_metrics(all_logits, all_labels)

    metrics    = compute_metrics(all_logits, all_labels)

    elapsed = time.time() - t0
    return {**avg_losses, **metrics, "time_s": elapsed}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main training routine
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train(cfg: TrainConfig = train_cfg, args: argparse.Namespace | None = None):
    # Command-line overrides
    if args is not None:
        if args.data_root:   cfg.data_root   = args.data_root
        if args.epochs:      cfg.n_epochs    = args.epochs
        if args.batch_size:  cfg.batch_size  = args.batch_size
        if args.device:      cfg.device      = args.device

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # â”€â”€ Dataloaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_loader, val_loader, test_loader = build_dataloaders(
        labels_csv    = cfg.labels_csv,
        data_map_json = cfg.data_map_json,
        data_root     = cfg.data_root,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
        val_split     = cfg.val_split,
        test_split    = cfg.test_split,
        seed          = cfg.seed,
    )
    print(f"[train] Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # ── Compute per-class pos_weight from the training set ─────────────────
    # This penalises missed predictions of rare classes, fixing the collapse
    # where the model always/never fires for certain classes.
    all_train_labels = torch.stack([train_loader.dataset[i]["label"]
                                    for i in range(len(train_loader.dataset))])
    pos_counts  = all_train_labels.sum(dim=0).clamp(min=1)           # (8,)
    neg_counts  = len(all_train_labels) - pos_counts
    pos_weight  = (neg_counts / pos_counts).to(device)
    print(f"[train] pos_weight (rare=high): {pos_weight.cpu().tolist()}")
    # ───────────────────────────────────────────────────────────────────────

    # â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    encoder = AudioToSpikes(cochlea_cfg, audio_cfg, enc_cfg).to(device)
    model   = SNNSoundSeparator(snn_cfg).to(device)
    print(f"[train] Encoder params:  {count_parameters(encoder):,}")
    print(f"[train] SNN model params:{count_parameters(model):,}")

    # â”€â”€ Loss & optimiser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    loss_fn   = SeparationLoss(
        lambda_ce=cfg.lambda_ce, lambda_sep=cfg.lambda_sep,
        pos_weight=pos_weight,
    )
    optimizer = AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    # ReduceLROnPlateau: halves LR if val F1 stops improving for 15 epochs
    # More robust than cosine for noisy small-dataset val curves
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6,
    )
    scaler    = GradScaler() if device.type == "cuda" else None

    # â”€â”€ Logging & checkpointing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    best_val_f1    = 0.0
    best_ckpt_path = Path(cfg.checkpoint_dir) / "best_model.pt"

    # â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for epoch in range(1, cfg.n_epochs + 1):
        train_stats = run_epoch(
            model, encoder, loss_fn, train_loader, optimizer, device, scaler, is_train=True
        )
        val_stats = run_epoch(
            model, encoder, loss_fn, val_loader, None, device, None, is_train=False
        )
        scheduler.step(val_stats['f1'])

        # â”€â”€ Log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lr_now = optimizer.param_groups[0]["lr"]
        for split, stats in [("train", train_stats), ("val", val_stats)]:
            for k, v in stats.items():
                if k != "time_s":
                    writer.add_scalar(f"{split}/{k}", v, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(
            f"Epoch {epoch:3d}/{cfg.n_epochs} | "
            f"Train loss={train_stats['total']:.4f}  F1={train_stats['f1']:.3f} | "
            f"Val loss={val_stats['total']:.4f}  F1={val_stats['f1']:.3f} | "
            f"LR={lr_now:.2e} | {train_stats['time_s']:.1f}s"
        )

        # â”€â”€ Checkpoint best â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if val_stats["f1"] > best_val_f1:
            best_val_f1 = val_stats["f1"]
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "encoder_state":encoder.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "val_f1":       best_val_f1,
                    "snn_cfg":      snn_cfg,
                    "audio_cfg":    audio_cfg,
                },
                best_ckpt_path,
            )
            print(f"  âœ“ Saved best checkpoint  (val F1={best_val_f1:.3f})")

    # ── Per-class threshold search on the full validation set ─────────────
    print("\n[train] Searching for optimal per-class thresholds on val set ...")
    val_logits_all, val_labels_all = [], []
    model.eval(); encoder.eval()
    with torch.no_grad():
        for batch in val_loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            labels_b = batch["label"].to(device, non_blocking=True)
            spikes   = encoder(waveform)
            out      = model(spikes)
            val_logits_all.append(out["logits"].cpu())
            val_labels_all.append(labels_b.cpu())
    val_logits_all = torch.cat(val_logits_all)
    val_labels_all = torch.cat(val_labels_all)
    best_thresholds = find_optimal_thresholds(val_logits_all, val_labels_all)
    # ─────────────────────────────────────────────────────────────────────

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n[train] Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    encoder.load_state_dict(ckpt["encoder_state"])

    # Standard evaluation (threshold=0.5)
    test_stats = run_epoch(
        model, encoder, loss_fn, test_loader, None, device, None, is_train=False
    )

    # Re-evaluate with tuned per-class thresholds
    test_logits_all, test_labels_all = [], []
    model.eval(); encoder.eval()
    with torch.no_grad():
        for batch in test_loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            labels_b = batch["label"].to(device, non_blocking=True)
            spikes   = encoder(waveform)
            out      = model(spikes)
            test_logits_all.append(out["logits"].cpu())
            test_labels_all.append(labels_b.cpu())
    test_logits_all = torch.cat(test_logits_all)
    test_labels_all = torch.cat(test_labels_all)
    tuned_metrics = compute_metrics(test_logits_all, test_labels_all,
                                    per_class_thresholds=best_thresholds)

    print(
        f"\n-- TEST RESULTS (threshold=0.5) --\n"
        f"  Loss:        {test_stats['total']:.4f}\n"
        f"  Precision:   {test_stats['precision']:.3f}\n"
        f"  Recall:      {test_stats['recall']:.3f}\n"
        f"  F1:          {test_stats['f1']:.3f}\n"
        f"  Exact-match: {test_stats['exact_match']:.3f}\n"
    )
    print(
        f"-- TEST RESULTS (tuned per-class thresholds) --\n"
        f"  Precision:   {tuned_metrics['precision']:.3f}\n"
        f"  Recall:      {tuned_metrics['recall']:.3f}\n"
        f"  F1:          {tuned_metrics['f1']:.3f}\n"
        f"  Exact-match: {tuned_metrics['exact_match']:.3f}\n"
        f"  Thresholds:  {[round(t, 2) for t in best_thresholds]}\n"
    )

    writer.close()
    return model, encoder


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLI entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN sound source separator")
    parser.add_argument("--data_root",  type=str,  default=None)
    parser.add_argument("--epochs",     type=int,  default=None)
    parser.add_argument("--batch_size", type=int,  default=None)
    parser.add_argument("--device",     type=str,  default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()
    train(args=args)