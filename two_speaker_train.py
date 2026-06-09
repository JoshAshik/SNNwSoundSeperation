"""
two_speaker_train.py — v3 training: 2-speaker PIT SNN-TasNet on FSD50K.

Architecture: SNNTasNet(n_speakers=2, snn_mode='snn')
Dataset:      TwoSpeakerMixDataset — on-the-fly mixing from FSD50K clips
Loss:         PIT SI-SDR + multi-resolution spectral + spike-rate + reconstruction
Resume:       --resume loads best_2spk_latest.pt (saved every epoch)
SLURM:        slurm_v3_twospeaker.sh (self-resubmitting, 6h wall time)

BPTT design
───────────
LIF neurons are stateful (membrane voltage U[t] depends on U[t-1]), so naively
unrolling over the full 3 s clip (48000 samples → 3000 encoder frames) would
build a 3000-step computational graph — too deep to backprop through and too
large to fit in 16 GB VRAM.

ChunkedSNNSeparator solves this by splitting the encoder output into independent
overlapping windows (default chunk_size=100 frames = 0.1 s). LIF membrane state
is RESET TO ZERO at the start of every chunk. This means:
  • BPTT depth = 100 frames (not 3000)
  • All chunks are processed as one large batched matmul (B*n_chunks, hidden)
  • Memory footprint scales with chunk_size × hidden, not T × hidden

No additional truncation is needed at the training-script level. Gradient
clipping (default max_norm=5.0) is applied as a belt-and-suspenders measure
against any spike-induced gradient spikes.

Usage (smoke test):
    python3 two_speaker_train.py --n_epochs 3 --scenes_per_epoch 40 \\
        --batch_size 4 --num_workers 0 --fsd50k_root ./data/fsd50k

Usage (ARC):
    sbatch slurm_v3_twospeaker.sh
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from sep_model             import SNNTasNet, transfer_encoder_decoder
from two_speaker_dataset   import build_two_speaker_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (all overridable via argparse)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # ── dataset ──────────────────────────────────────────────────────────────
    fsd50k_root       = "./data/fsd50k",
    scenes_per_epoch  = 5_000,
    val_scenes        = 1_000,
    overlap_range     = (0.3, 1.0),
    snr_range         = (-5.0, 5.0),
    # ── model ─────────────────────────────────────────────────────────────────
    n_filters         = 256,
    kernel_sz         = 32,
    stride            = 16,
    hidden            = 256,
    n_layers          = 3,
    n_speakers        = 2,
    dropout           = 0.3,
    snn_chunk         = 100,
    # ── training ──────────────────────────────────────────────────────────────
    n_epochs          = 200,
    batch_size        = 8,
    num_workers       = 8,
    lr                = 1.5e-4,
    lr_finetune       = 5e-5,
    weight_decay      = 1e-4,
    freeze_epochs     = 20,
    val_every         = 5,
    grad_clip         = 5.0,
    ema_decay         = 0.999,
    seed              = 42,
    # ── loss ──────────────────────────────────────────────────────────────────
    lambda_spec            = 0.5,
    lambda_rate            = 0.10,
    lambda_recon           = 10.0,   # v6: raised from 5.0 — recon error was 0.75 at 5.0; need < 0.10
    target_spike_rate      = 0.10,
    target_spike_rate_max  = 0.30,   # v6: widened from 0.20 — model settles at ~0.25 in training
    lambda_balance         = 0.10,   # v6: channel energy balance loss (prevents speaker collapse)
    use_weight_norm        = True,   # v5: weight_norm on decoder (training stabiliser)
    # ── paths ─────────────────────────────────────────────────────────────────
    pretrain_ckpt     = "./checkpoints_pretrain/best_pretrain.pt",
    ckpt_dir          = "./checkpoints_v6",
    log_dir           = "./runs_v6",
    csv_path          = "./v6_train_log.csv",
    # ── runtime ───────────────────────────────────────────────────────────────
    resume            = False,
    max_wall_hours    = 0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMA  — identical implementation to snn_finetune.py
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.
    apply_shadow() / restore() swap data pointers (no tensor copies at eval time).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model   = model
        self.decay   = decay
        self.shadow  = {n: p.data.clone() for n, p in model.named_parameters()}
        self._backup = {}

    def update(self) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                self.shadow[n].lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self) -> None:
        self._backup = {}
        for n, p in self.model.named_parameters():
            self._backup[n] = p.data
            p.data = self.shadow[n]

    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            p.data = self._backup[n]

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = sd["shadow"]
        self.decay  = sd["decay"]


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant SDR (per sample, with gradients).
    Inputs: (B, T). Returns: (B,).
    """
    eps = 1e-8
    ref  = ref - ref.mean(dim=-1, keepdim=True)
    est  = est - est.mean(dim=-1, keepdim=True)
    dot  = (est * ref).sum(dim=-1)
    rp2  = (ref * ref).sum(dim=-1).clamp(min=eps)
    proj = (dot / rp2).unsqueeze(-1) * ref
    noise = est - proj
    ratio = proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp(min=eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps))   # (B,)


def _mr_spectral_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    fft_sizes: tuple = (256, 512, 1024),
) -> torch.Tensor:
    """
    Multi-resolution L1 loss on log-magnitude STFT.
    Inputs: (B, T). Returns: scalar.
    Computed at FFT sizes 256/512/1024 (same as SeparationLoss in sep_losses.py).
    """
    total = est.new_zeros(1).squeeze()
    B, T  = est.shape
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=est.device)
        kw  = dict(n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True, center=True)
        S_e = torch.stft(est.reshape(B, T), **kw).abs().clamp(min=1e-8).log()
        S_r = torch.stft(ref.reshape(B, T), **kw).abs().clamp(min=1e-8).log()
        total = total + F.l1_loss(S_e, S_r)
    return total / len(fft_sizes)


def _spike_rate_loss(
    spike_recs: list,
    target_min: float = 0.10,
    target_max: float = 0.20,
) -> torch.Tensor:
    """
    Per-layer L2 penalty driving each LIF layer's mean firing rate into
    [target_min, target_max].  No penalty when the rate is inside the band.

    v3 used mean L1 over all layers — a layer saturating at 0.52 could be
    hidden by other layers near target, producing a small averaged penalty.
    Per-layer L2 makes each saturating layer pay its own quadratic cost:
      rate=0.52, target_max=0.20 → excess=0.32 → L2=0.1024 per layer.

    Returns scalar tensor on the same device as spike records.
    """
    tensors = [s for s in spike_recs if isinstance(s, torch.Tensor)]
    if not tensors:
        return torch.tensor(0.0)
    total = tensors[0].new_zeros(1).squeeze()
    for s in tensors:
        rate   = s.float().mean()
        excess = F.relu(rate - target_max) + F.relu(target_min - rate)
        total  = total + excess.pow(2)
    return total   # sum of per-layer L2 penalties (not averaged)


def _recon_loss(estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """
    MSE energy conservation loss: (est_ch0 + est_ch1) should equal mixture.
    estimates: (B, 2, T), mixture: (B, T).

    v3/v4 used scale-normalised L1 (divided by mixture.abs().mean()).
    That formulation has inconsistent gradient magnitude across batches and
    allowed the decoder to sustain a 7× amplitude offset (recon_error=5.1).

    v5: RMS-normalised MSE.  For a 7× amplification:
      MSE = mean((7x - x)^2) / mean(x^2) = 36  (scale-free, batch-consistent)
    With lambda_recon=5.0 this contributes 180 to the total loss — strong
    enough to compete with the SI-SDR gradient even for large amplitude errors.

    Normalising by RMS (not mean-abs) gives consistent gradient regardless of
    whether the mixture is loud or quiet.
    """
    summed    = estimates.sum(dim=1)                                    # (B, T)
    mix_power = (mixture ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-8)   # (B, 1) RMS²
    return ((summed - mixture) ** 2 / mix_power).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PIT forward pass — core of the 2-speaker training step
# ─────────────────────────────────────────────────────────────────────────────

def pit_forward(
    model:    SNNTasNet,
    mixture:  torch.Tensor,   # (B, T)
    t1:       torch.Tensor,   # (B, T) — reference for channel 0
    t2:       torch.Tensor,   # (B, T) — reference for channel 1
    cfg,
) -> tuple:
    """
    Run the model forward pass and compute the PIT loss + metrics.

    PIT (Permutation Invariant Training) for 2 speakers:
      Permutation 0: est[:,0] → t1, est[:,1] → t2
      Permutation 1: est[:,0] → t2, est[:,1] → t1

    The per-sample negative SI-SDR is computed for both permutations.
    torch.min is differentiable — gradient flows only through the winning
    (lower-loss) permutation. The losing permutation receives zero gradient.

    Auxiliary losses (spectral, spike-rate, reconstruction) are computed using
    the winning permutation's assignment so every loss term is aligned.

    Returns: (total_loss, metrics_dict)
    """
    B, T = mixture.shape

    # ── Model forward ─────────────────────────────────────────────────────────
    out = model(mixture)           # {"separated": (B, 2, T'), "masks": ..., "spike_recs": ...}
    est = out["separated"]         # (B, 2, T')

    # Trim ConvTranspose1d overshoot (decoder adds ~stride samples at boundaries)
    T_out = est.shape[-1]
    if T_out > T:
        est = est[..., :T]
    elif T_out < T:
        est = F.pad(est, (0, T - T_out))

    # ── PIT selection via per-sample negative SI-SDR ───────────────────────────
    # Compute SI-SDR (higher = better) for each permutation, per sample.
    sdr_p0 = (_si_sdr(est[:, 0], t1) + _si_sdr(est[:, 1], t2)) / 2   # (B,)
    sdr_p1 = (_si_sdr(est[:, 0], t2) + _si_sdr(est[:, 1], t1)) / 2   # (B,)

    # Stack as negative SDR (lower = better), then min over permutations.
    # torch.min is differentiable: gradient is 1 for the winner, 0 for the loser.
    neg_sdr_stack = torch.stack([-sdr_p0, -sdr_p1], dim=1)            # (B, 2)
    best_neg, best_perm = neg_sdr_stack.min(dim=1)                     # (B,), (B,)
    si_sdr_loss = best_neg.mean()

    # ── Gather aligned estimates for auxiliary losses ──────────────────────────
    # best_perm[b] == 0 → est[b,0] matched t1, est[b,1] matched t2
    # best_perm[b] == 1 → est[b,0] matched t2, est[b,1] matched t1  (flip)
    #
    # idx_t1[b] is the channel index of the estimate matched to t1 for sample b.
    # When perm=0: est[:,0] matches t1 → idx=0
    # When perm=1: est[:,1] matches t1 → idx=1
    # idx_t1 == best_perm  ✓
    idx_t1 = best_perm.view(B, 1, 1).expand(B, 1, T).long()
    idx_t2 = (1 - best_perm).view(B, 1, 1).expand(B, 1, T).long()
    est_t1 = est.gather(1, idx_t1).squeeze(1)    # (B, T) — matched to t1
    est_t2 = est.gather(1, idx_t2).squeeze(1)    # (B, T) — matched to t2

    # ── Auxiliary losses ───────────────────────────────────────────────────────
    spike_recs = out.get("spike_recs", [])
    rate_loss  = _spike_rate_loss(spike_recs, cfg.target_spike_rate, cfg.target_spike_rate_max)
    spec_loss  = (_mr_spectral_loss(est_t1, t1) + _mr_spectral_loss(est_t2, t2)) / 2
    recon_loss = _recon_loss(est, mixture)

    # Channel energy balance loss (v6): penalises speaker collapse.
    # If one channel absorbs all energy, ch_rms[:,0]/ch_rms[:,1] → 0 or ∞;
    # log().pow(2) is zero when both channels carry equal energy and grows
    # symmetrically as the ratio diverges in either direction.
    # Works alongside PIT — PIT picks the best permutation; this term ensures
    # neither channel is silenced regardless of which permutation wins.
    ch_rms       = est.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)   # (B, 2)
    balance_loss = (ch_rms[:, 0] / ch_rms[:, 1]).log().pow(2).mean()

    total = (si_sdr_loss
             + cfg.lambda_spec    * spec_loss
             + cfg.lambda_rate    * (rate_loss if isinstance(rate_loss, torch.Tensor)
                                     else torch.tensor(rate_loss, device=si_sdr_loss.device))
             + cfg.lambda_recon   * recon_loss
             + cfg.lambda_balance * balance_loss)

    # ── Metrics (no grad) ──────────────────────────────────────────────────────
    with torch.no_grad():
        best_sdr    = -best_neg                                          # (B,)
        # SI-SDRi baseline: mixture vs same refs as the winning permutation.
        # Since (sdr(mix,t1)+sdr(mix,t2))/2 is symmetric to permutation choice,
        # we can always average over both refs for the mixture baseline.
        mix_sdr_avg = (_si_sdr(mixture, t1) + _si_sdr(mixture, t2)) / 2  # (B,)
        si_sdri_val = (best_sdr - mix_sdr_avg).mean().item()

    spike_tensors = [s for s in spike_recs if isinstance(s, torch.Tensor)]
    mean_spike = (
        sum(s.float().mean().item() for s in spike_tensors) / len(spike_tensors)
        if spike_tensors else 0.0
    )

    metrics = {
        "loss":       total.item(),
        "si_sdri":    si_sdri_val,
        "spec_loss":  spec_loss.item(),
        "rate_loss":  rate_loss.item() if isinstance(rate_loss, torch.Tensor) else 0.0,
        "recon_loss": recon_loss.item(),
        "spike_rate": mean_spike,
        "wform_mean": est.abs().mean().item(),
    }
    return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freeze_encoder_decoder(model: SNNTasNet) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(False)


def _unfreeze_all(model: SNNTasNet) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save helper (atomic rename — safe on BeeGFS)
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(
    path:             Path,
    epoch:            int,
    model:            SNNTasNet,
    ema:              EMA,
    optimizer,
    scheduler,
    val_si_sdri:      float,
    best_val_si_sdri: float,
    cfg,
) -> None:
    """Write to a .tmp file then atomically rename to the target path."""
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "ema_state":        ema.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "val_si_sdri":      val_si_sdri,
        "best_val_si_sdri": best_val_si_sdri,
        "model_cfg": {
            "n_filters":       cfg.n_filters,   "kernel_sz":       cfg.kernel_sz,
            "stride":          cfg.stride,       "hidden":          cfg.hidden,
            "n_layers":        cfg.n_layers,     "dropout":         cfg.dropout,
            "snn_mode":        "snn",            "snn_chunk":       cfg.snn_chunk,
            "n_speakers":      cfg.n_speakers,   "use_weight_norm": cfg.use_weight_norm,
        },
    }, tmp)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:    SNNTasNet,
    loader,
    optimizer,
    device:   torch.device,
    scaler,
    cfg,
    is_train: bool,
    ema:      EMA = None,
) -> dict:
    """
    Run one full pass over `loader`.

    Train mode:  computes PIT loss, backprops, clips gradients, steps optimizer,
                 updates EMA shadow weights.
    Eval mode:   runs under EMA weights (apply_shadow called by caller before
                 invoking this function), no gradients computed.

    Returns a dict of mean metrics over all batches.
    """
    model.train(is_train)
    totals: dict = {}
    n_batches    = 0
    t_data       = time.time()

    for batch_idx, (mixture, t1, t2) in enumerate(loader):
        mixture = mixture.to(device, non_blocking=True)   # (B, T)
        t1      = t1.to(device, non_blocking=True)
        t2      = t2.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            loss, metrics = pit_forward(model, mixture, t1, t2, cfg)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip,
                )
                optimizer.step()

            if ema is not None:
                ema.update()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        if is_train and batch_idx % 100 == 0 and batch_idx > 0:
            t_batch = time.time() - t_data
            print(
                f"  [batch {batch_idx:>4}/{len(loader)}] "
                f"loss={metrics['loss']:.4f}  "
                f"SI-SDRi={metrics['si_sdri']:>+.2f} dB  "
                f"spike={metrics['spike_rate']:.3f}  "
                f"{t_batch:.1f}s",
                flush=True,
            )
            t_data = time.time()

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v3 2-speaker PIT SNN-TasNet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── flags referenced in slurm_v3_twospeaker.sh ────────────────────────────
    p.add_argument("--resume",            action="store_true")
    p.add_argument("--max_wall_hours",    type=float, default=defaults["max_wall_hours"],
                   help="Exit cleanly after this many hours (0 = no limit)")
    p.add_argument("--n_epochs",          type=int,   default=defaults["n_epochs"])
    p.add_argument("--batch_size",        type=int,   default=defaults["batch_size"])
    p.add_argument("--num_workers",       type=int,   default=defaults["num_workers"])
    p.add_argument("--scenes_per_epoch",  type=int,   default=defaults["scenes_per_epoch"])
    p.add_argument("--val_every",         type=int,   default=defaults["val_every"])
    p.add_argument("--fsd50k_root",       type=str,   default=defaults["fsd50k_root"])
    p.add_argument("--ckpt_dir",          type=str,   default=defaults["ckpt_dir"])
    # ── optional hyperparameter overrides ─────────────────────────────────────
    p.add_argument("--pretrain_ckpt",     type=str,   default=defaults["pretrain_ckpt"])
    p.add_argument("--hidden",            type=int,   default=defaults["hidden"])
    p.add_argument("--n_layers",          type=int,   default=defaults["n_layers"])
    p.add_argument("--freeze_epochs",     type=int,   default=defaults["freeze_epochs"])
    p.add_argument("--lr",                type=float, default=defaults["lr"])
    p.add_argument("--lr_finetune",       type=float, default=defaults["lr_finetune"])
    p.add_argument("--lambda_rate",       type=float, default=defaults["lambda_rate"])
    p.add_argument("--lambda_spec",       type=float, default=defaults["lambda_spec"])
    p.add_argument("--lambda_recon",      type=float, default=defaults["lambda_recon"])
    p.add_argument("--grad_clip",         type=float, default=defaults["grad_clip"])
    p.add_argument("--seed",              type=int,   default=defaults["seed"])
    p.add_argument("--log_dir",           type=str,   default=defaults["log_dir"])
    p.add_argument("--csv_path",          type=str,   default=defaults["csv_path"])
    p.add_argument("--lambda_balance",    type=float, default=defaults["lambda_balance"])

    cfg = p.parse_args()
    for k, v in defaults.items():         # back-fill any defaults not in argparse
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args(DEFAULTS)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cpu":
        print("[warn] No GPU — SNN training on CPU is very slow")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("\n[data] Building 2-speaker FSD50K dataloaders ...")
    train_loader, val_loader = build_two_speaker_dataloaders(
        fsd50k_root      = cfg.fsd50k_root,
        scenes_per_epoch = cfg.scenes_per_epoch,
        val_scenes       = cfg.val_scenes,
        batch_size       = cfg.batch_size,
        num_workers      = cfg.num_workers,
        seed             = cfg.seed,
    )
    print(f"[data] train={len(train_loader)} batches/ep  "
          f"val={len(val_loader)} batches/ep")

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_dir    = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt   = ckpt_dir / "best_2spk.pt"
    latest_ckpt = ckpt_dir / "best_2spk_latest.pt"
    _peek_ckpt  = latest_ckpt if latest_ckpt.exists() else best_ckpt

    # ── Architecture peek (read saved cfg BEFORE building model) ──────────────
    # Prevents shape mismatch if DEFAULTS change between runs.
    if cfg.resume and _peek_ckpt.exists():
        _early = torch.load(_peek_ckpt, map_location="cpu", weights_only=False)
        _saved = _early.get("model_cfg", {})
        for k in ("n_filters", "kernel_sz", "stride", "hidden",
                  "n_layers", "dropout", "snn_chunk"):
            if k in _saved:
                setattr(cfg, k, _saved[k])
        print(f"[resume] Architecture from checkpoint: "
              f"hidden={cfg.hidden}  n_layers={cfg.n_layers}")
        del _early, _saved

    # ── Build model ───────────────────────────────────────────────────────────
    model = SNNTasNet(
        n_filters       = cfg.n_filters,
        kernel_sz       = cfg.kernel_sz,
        stride          = cfg.stride,
        hidden          = cfg.hidden,
        n_layers        = cfg.n_layers,
        n_speakers      = cfg.n_speakers,
        dropout         = cfg.dropout,
        snn_mode        = "snn",
        snn_chunk       = cfg.snn_chunk,
        use_weight_norm = cfg.use_weight_norm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    bptt_s   = cfg.snn_chunk * cfg.stride / 16_000
    print(f"\n[model] SNNTasNet  n_speakers={cfg.n_speakers}  params={n_params:,}")
    print(f"        hidden={cfg.hidden}  n_layers={cfg.n_layers}  "
          f"chunk={cfg.snn_chunk} frames ({bptt_s:.2f}s BPTT window)")
    print(f"        LIF membrane reset at chunk boundaries → "
          f"BPTT depth={cfg.snn_chunk} (not {cfg.stride * cfg.snn_chunk})")

    # ── Transfer pre-trained encoder/decoder (WHAM! Stage 1) ─────────────────
    pretrain_path = Path(cfg.pretrain_ckpt)
    if pretrain_path.exists():
        print(f"\n[transfer] Loading pre-trained encoder/decoder from {pretrain_path}")
        pt      = torch.load(pretrain_path, map_location=device, weights_only=False)
        src_cfg = pt.get("model_cfg", {})
        src     = SNNTasNet(
            n_filters = src_cfg.get("n_filters", cfg.n_filters),
            kernel_sz = src_cfg.get("kernel_sz", cfg.kernel_sz),
            stride    = src_cfg.get("stride",    cfg.stride),
            hidden    = src_cfg.get("hidden",    cfg.hidden),
            n_layers  = src_cfg.get("n_layers",  cfg.n_layers),
            snn_mode  = "gru",
        ).to(device)
        if "ema_state" in pt and pt["ema_state"].get("shadow"):
            for n, p in src.named_parameters():
                if n in pt["ema_state"]["shadow"]:
                    p.data.copy_(pt["ema_state"]["shadow"][n])
        else:
            src.load_state_dict(pt["model_state"], strict=False)
        transfer_encoder_decoder(src, model)
        print(f"[transfer] Pre-train val SI-SDRi: "
              f"{pt.get('val_si_sdri', float('nan')):+.2f} dB")
        del src, pt
    else:
        print(f"\n[warn] No pre-train checkpoint at {pretrain_path}. "
              "Training encoder from scratch — slower convergence expected.")

    # ── Freeze phase: determine from checkpoint epoch ─────────────────────────
    resume_past_freeze = False
    if cfg.resume and _peek_ckpt.exists():
        _pk = torch.load(_peek_ckpt, map_location="cpu", weights_only=False)
        resume_past_freeze = (_pk.get("epoch", 0) + 1) > cfg.freeze_epochs
        del _pk
    unfrozen = resume_past_freeze

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if unfrozen:
        _unfreeze_all(model)
        optimizer = AdamW([
            {"params": model.separator.parameters(), "lr": cfg.lr},
            {"params": list(model.encoder.parameters()) +
                       list(model.decoder.parameters()), "lr": cfg.lr_finetune},
        ], weight_decay=cfg.weight_decay)
        print(f"[optim] Two-group AdamW (finetune phase restored)")
    else:
        _freeze_encoder_decoder(model)
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        print(f"[optim] Single-group AdamW  lr={cfg.lr:.1e}  "
              f"(encoder frozen for first {cfg.freeze_epochs} epochs)")

    # Single cosine decay over post-freeze epochs — no warm restarts.
    # Warm restarts repeatedly reset LR and destroyed the v1 run at ep700.
    # See docs/decisions.md §CosineAnnealingLR.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = max(1, cfg.n_epochs - cfg.freeze_epochs),
        eta_min= 1e-6,
    )

    ema    = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = -1e9
    if cfg.resume and _peek_ckpt.exists():
        print(f"\n[resume] Loading {_peek_ckpt}")
        ckpt = torch.load(_peek_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "ema_state"       in ckpt: ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val_si_sdri", ckpt.get("val_si_sdri", -1e9))
        print(f"[resume] Epoch {start_epoch}  best={best_val:+.2f} dB  "
              f"phase={'finetune' if unfrozen else 'freeze'}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = Path(cfg.csv_path)
    _header  = ["epoch", "phase", "trn_loss", "trn_si_sdri", "val_si_sdri",
                 "gap", "spike_rate", "recon_loss", "spec_loss",
                 "lr", "epoch_time_s", "is_best"]
    if not cfg.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_header)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    job_start   = time.time()
    epoch_times: deque = deque(maxlen=10)
    vl: dict = {}

    sep = "─" * 112
    print(f"\n{sep}")
    print(f"{'Ep':>5}  {'Phase':>7}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'Spike':>7}  {'LR':>8}  {'t':>5}  {'ETA':>8}")
    print(sep)

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()

        # ── Unfreeze encoder after freeze phase ───────────────────────────────
        if epoch == cfg.freeze_epochs + 1 and not unfrozen:
            print(f"\n  [epoch {epoch}] Unfreezing encoder+decoder  "
                  f"lr_finetune={cfg.lr_finetune:.1e}")
            _unfreeze_all(model)
            optimizer = AdamW([
                {"params": model.separator.parameters(), "lr": cfg.lr},
                {"params": list(model.encoder.parameters()) +
                           list(model.decoder.parameters()), "lr": cfg.lr_finetune},
            ], weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg.n_epochs - epoch), eta_min=1e-6,
            )
            unfrozen = True

        phase = "freeze" if epoch <= cfg.freeze_epochs else "finetune"

        # ── Train ─────────────────────────────────────────────────────────────
        tr = run_epoch(model, train_loader, optimizer, device, scaler,
                       cfg, is_train=True, ema=ema)

        # ── Validate (every val_every epochs and on the final epoch) ──────────
        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.n_epochs)
        if do_val:
            ema.apply_shadow()
            vl = run_epoch(model, val_loader, None, device, None,
                           cfg, is_train=False, ema=None)
            ema.restore()

        scheduler.step()

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        epoch_times.append(dt)
        avg_t  = sum(epoch_times) / len(epoch_times)
        remain = avg_t * (cfg.n_epochs - epoch)
        eta    = f"{int(remain//3600)}h{int((remain%3600)//60):02d}m"
        gap    = tr["si_sdri"] - vl.get("si_sdri", float("nan"))

        if do_val:
            val_str = (f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB  "
                       f"{vl['spike_rate']:>7.4f}")
        else:
            val_str = f"{'---':>9}    {'---':>6}   {'---':>7}"

        print(f"{epoch:>5}/{cfg.n_epochs}  {phase:>7}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{val_str}  {lr:.2e}  {dt:>4.1f}s  {eta:>8}")

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalar("si_sdri/train",  tr["si_sdri"],  epoch)
        writer.add_scalar("loss/train",     tr["loss"],     epoch)
        writer.add_scalar("spike/train",    tr["spike_rate"], epoch)
        writer.add_scalar("lr",             lr,             epoch)
        if do_val:
            writer.add_scalar("si_sdri/val",    vl["si_sdri"],    epoch)
            writer.add_scalar("gap/train_val",  gap,              epoch)
            writer.add_scalar("spike/val",      vl["spike_rate"], epoch)
            writer.add_scalar("recon_loss/val", vl["recon_loss"], epoch)

        # ── Checkpoint: best_2spk.pt (only on improvement) ───────────────────
        is_best = do_val and vl["si_sdri"] > best_val
        if is_best:
            best_val = vl["si_sdri"]
            _save_ckpt(best_ckpt, epoch, model, ema, optimizer, scheduler,
                       best_val, best_val, cfg)
            print(f"  ✔ New best  val={best_val:+.2f} dB  gap={gap:+.1f} dB")

        # ── Checkpoint: best_2spk_latest.pt (every epoch — SLURM resume) ─────
        _save_ckpt(latest_ckpt, epoch, model, ema, optimizer, scheduler,
                   vl.get("si_sdri", float("nan")), best_val, cfg)

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase,
                round(tr["loss"],        4), round(tr["si_sdri"],  4),
                round(vl["si_sdri"],     4) if do_val else "",
                round(gap,               4) if do_val else "",
                round(vl.get("spike_rate", 0.0), 4) if do_val else "",
                round(vl.get("recon_loss",  0.0), 4) if do_val else "",
                round(vl.get("spec_loss",   0.0), 4) if do_val else "",
                lr, round(dt, 1), int(is_best),
            ])

        # ── Wall-time guard (for SLURM 6h limit) ─────────────────────────────
        if cfg.max_wall_hours > 0:
            elapsed = time.time() - job_start
            if elapsed >= cfg.max_wall_hours * 3600:
                print(f"\n[wall-time] {elapsed/3600:.2f}h elapsed — "
                      f"checkpoint at {latest_ckpt}")
                print(f"[wall-time] Resubmit with --resume to continue "
                      f"from epoch {epoch + 1}.")
                writer.close()
                return

        # ── Early diagnostic ─────────────────────────────────────────────────
        # If epoch 3 (first non-val epoch we can reasonably check) shows silent
        # outputs, the fsd50k_root path is probably wrong.
        if epoch == 3 and vl.get("wform_mean", 1.0) < 1e-4:
            print("\n  ⚠ WARNING: waveform_mean ≈ 0 after 3 epochs. "
                  "Check --fsd50k_root path.\n")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[done] Training complete.  "
          f"Best val SI-SDRi = {best_val:+.2f} dB")
    print(f"       Best checkpoint : {best_ckpt}")
    print(f"       CSV log         : {csv_path}")
    writer.close()


if __name__ == "__main__":
    main()
