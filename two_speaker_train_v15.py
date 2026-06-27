"""
two_speaker_train_v15.py — v15 dynamic clean mixing + speed perturbation.

Architecture: SNNTasNet v15 (sep_model_v10.py, snn_mode=dprnn)
  Same DPRNNSeparator as v13 (rnn_hidden=128, bn_dim=64, 6 blocks).
  v14 showed wider rnn_hidden=256 didn't help — data diversity is the bottleneck.

v15 changes vs v13/v14:
  - Dynamic on-the-fly mixing (dynamic_mix_dataset.py) instead of fixed 13,900 pairs
  - Speed perturbation 0.95x–1.05x on each source independently
  - Random relative SNR ±5 dB between sources
  - Effectively infinite training combinations per epoch
  - Architecture back to v13 size (rnn_hidden=128) — proven config
  - Warmstart encoder+decoder from v13 best

Dataset:   DynamicMixDataset — on-the-fly mixing from Libri2Mix clean sources
Val:       Pre-generated LibriMixDataset (dev split) — fair comparison
Resume:    --resume loads checkpoints_v15/best_2spk_latest.pt
Warmstart: --warmstart (default: checkpoints_v13/best_2spk.pt)
SLURM:     slurm_v15_twospeaker.sh (self-resubmitting, 6 h wall)

Usage (smoke test):
    python3 two_speaker_train_v15.py --n_epochs 2 --batch_size 2 --num_workers 0 \\
        --librimix_root ./data/librimix --max_wall_hours 0

Usage (ARC):
    sbatch slurm_v15_twospeaker.sh
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

from sep_model_v10       import SNNTasNet
from dynamic_mix_dataset import build_dynamic_mix_dataloaders, DEFAULT_CLIP_LEN


# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (v15 dynamic mixing, v13-sized architecture)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # ── dataset ──────────────────────────────────────────────────────────────
    librimix_root     = "/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max",
    train_split       = "train-100",
    val_split         = "dev",
    clip_len          = DEFAULT_CLIP_LEN,
    epoch_size        = 13_900,
    speed_perturb     = True,
    snr_min_db        = -5.0,
    snr_max_db        = 5.0,
    # ── model ─────────────────────────────────────────────────────────────────
    n_filters         = 256,
    kernel_sz         = 32,
    stride            = 16,
    hidden            = 128,       # back to v13 size (v14 wider didn't help)
    n_layers          = 6,
    n_speakers        = 2,
    dropout           = 0.1,
    snn_chunk         = 200,
    decoder_refine    = 3,
    decoder_groups    = 8,
    dprnn_bn_dim      = 64,
    dprnn_rnn_hidden  = 128,       # back to v13 size
    # ── training ──────────────────────────────────────────────────────────────
    n_epochs          = 200,
    batch_size        = 16,
    num_workers       = 8,
    lr                = 5e-4,
    lr_finetune       = 5e-5,
    weight_decay      = 1e-4,
    freeze_epochs     = 20,
    val_every         = 5,
    grad_clip         = 5.0,
    ema_decay         = 0.999,
    seed              = 42,
    # ── augmentation ──────────────────────────────────────────────────────────
    gain_aug_db       = 3.0,
    train_augment     = True,
    # ── loss ──────────────────────────────────────────────────────────────────
    lambda_spec           = 0.5,
    lambda_rate           = 0.0,
    lambda_recon          = 5.0,
    target_spike_rate     = 0.10,
    target_spike_rate_max = 0.30,
    # ── paths ─────────────────────────────────────────────────────────────────
    warmstart         = "./checkpoints_v13/best_2spk.pt",
    starter_ckpt      = "./checkpoints_v13/best_2spk.pt",
    ckpt_dir          = "./checkpoints_v15",
    log_dir           = "./runs_v15",
    csv_path          = "./v15_train_log.csv",
    # ── runtime ───────────────────────────────────────────────────────────────
    resume            = False,
    max_wall_hours    = 0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
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
# Per-source gain augmentation  (training only)
# ─────────────────────────────────────────────────────────────────────────────

def apply_gain_aug(
    t1: torch.Tensor,
    t2: torch.Tensor,
    gain_db: float,
) -> tuple:
    B = t1.shape[0]
    g1 = 10.0 ** (t1.new_empty(B, 1).uniform_(-gain_db, gain_db) / 20.0)
    g2 = 10.0 ** (t2.new_empty(B, 1).uniform_(-gain_db, gain_db) / 20.0)
    t1_aug  = t1 * g1
    t2_aug  = t2 * g2
    mix_aug = t1_aug + t2_aug
    return mix_aug, t1_aug, t2_aug


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    eps  = 1e-8
    ref  = ref - ref.mean(dim=-1, keepdim=True)
    est  = est - est.mean(dim=-1, keepdim=True)
    dot  = (est * ref).sum(dim=-1)
    rp2  = (ref * ref).sum(dim=-1).clamp(min=eps)
    proj = (dot / rp2).unsqueeze(-1) * ref
    noise = est - proj
    ratio = proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp(min=eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps))


def _mr_spectral_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    fft_sizes: tuple = (256, 512, 1024),
) -> torch.Tensor:
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
    target_max: float = 0.30,
) -> torch.Tensor:
    tensors = [s for s in spike_recs if isinstance(s, torch.Tensor)]
    if not tensors:
        return torch.tensor(0.0)
    total = tensors[0].new_zeros(1).squeeze()
    for s in tensors:
        rate   = s.float().mean()
        excess = F.relu(rate - target_max) + F.relu(target_min - rate)
        total  = total + excess.pow(2)
    return total


def _recon_loss(estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    summed    = estimates.sum(dim=1)
    mix_power = (mixture ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return ((summed - mixture) ** 2 / mix_power).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PIT forward pass
# ─────────────────────────────────────────────────────────────────────────────

def pit_forward(
    model:   SNNTasNet,
    mixture: torch.Tensor,
    t1:      torch.Tensor,
    t2:      torch.Tensor,
    cfg,
) -> tuple:
    B, T = mixture.shape
    out  = model(mixture)
    est  = out["separated"]

    T_out = est.shape[-1]
    if T_out > T:
        est = est[..., :T]
    elif T_out < T:
        est = F.pad(est, (0, T - T_out))

    sdr_p0 = (_si_sdr(est[:, 0], t1) + _si_sdr(est[:, 1], t2)) / 2
    sdr_p1 = (_si_sdr(est[:, 0], t2) + _si_sdr(est[:, 1], t1)) / 2

    neg_sdr_stack       = torch.stack([-sdr_p0, -sdr_p1], dim=1)
    best_neg, best_perm = neg_sdr_stack.min(dim=1)
    si_sdr_loss         = best_neg.mean()

    idx_t1  = best_perm.view(B, 1, 1).expand(B, 1, T).long()
    idx_t2  = (1 - best_perm).view(B, 1, 1).expand(B, 1, T).long()
    est_t1  = est.gather(1, idx_t1).squeeze(1)
    est_t2  = est.gather(1, idx_t2).squeeze(1)

    spike_recs = out.get("spike_recs", [])
    rate_loss  = _spike_rate_loss(spike_recs, cfg.target_spike_rate, cfg.target_spike_rate_max)
    spec_loss  = (_mr_spectral_loss(est_t1, t1) + _mr_spectral_loss(est_t2, t2)) / 2
    recon_loss = _recon_loss(est, mixture)

    rate_t = (rate_loss.to(si_sdr_loss.device) if isinstance(rate_loss, torch.Tensor)
              else torch.tensor(rate_loss, device=si_sdr_loss.device))

    total = (si_sdr_loss
             + cfg.lambda_spec  * spec_loss
             + cfg.lambda_rate  * rate_t
             + cfg.lambda_recon * recon_loss)

    with torch.no_grad():
        best_sdr    = -best_neg
        mix_sdr_avg = (_si_sdr(mixture, t1) + _si_sdr(mixture, t2)) / 2
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
        "rate_loss":  rate_t.item(),
        "recon_loss": recon_loss.item(),
        "spike_rate": mean_spike,
        "wform_mean": est.abs().mean().item(),
    }
    return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(path, epoch, model, ema, optimizer, scheduler,
               val_si_sdri, best_val_si_sdri, cfg) -> None:
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
            "n_filters":       cfg.n_filters,
            "kernel_sz":       cfg.kernel_sz,
            "stride":          cfg.stride,
            "hidden":          cfg.hidden,
            "n_layers":        cfg.n_layers,
            "dropout":         cfg.dropout,
            "snn_mode":        "dprnn",
            "snn_chunk":       cfg.snn_chunk,
            "n_speakers":      cfg.n_speakers,
            "decoder_refine":  cfg.decoder_refine,
            "decoder_groups":  cfg.decoder_groups,
            "use_weight_norm": False,
            "dprnn_bn_dim":    cfg.dprnn_bn_dim,
            "dprnn_rnn_hidden": cfg.dprnn_rnn_hidden,
        },
    }, tmp)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, scaler, cfg,
              is_train: bool, ema: EMA = None) -> dict:
    model.train(is_train)
    totals: dict = {}
    n_batches    = 0
    t_data       = time.time()

    for batch_idx, (mixture, t1, t2) in enumerate(loader):
        mixture = mixture.to(device, non_blocking=True)
        t1      = t1.to(device, non_blocking=True)
        t2      = t2.to(device, non_blocking=True)

        if is_train and cfg.gain_aug_db > 0.0:
            mixture, t1, t2 = apply_gain_aug(t1, t2, cfg.gain_aug_db)

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
        description="v15 dynamic clean mixing + speed perturbation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--warmstart",       type=str, default=defaults["warmstart"])
    p.add_argument("--max_wall_hours",  type=float, default=defaults["max_wall_hours"])
    p.add_argument("--n_epochs",        type=int,   default=defaults["n_epochs"])
    p.add_argument("--batch_size",      type=int,   default=defaults["batch_size"])
    p.add_argument("--num_workers",     type=int,   default=defaults["num_workers"])
    p.add_argument("--val_every",       type=int,   default=defaults["val_every"])
    p.add_argument("--librimix_root",   type=str,   default=defaults["librimix_root"])
    p.add_argument("--train_split",     type=str,   default=defaults["train_split"])
    p.add_argument("--val_split",       type=str,   default=defaults["val_split"])
    p.add_argument("--clip_len",        type=int,   default=defaults["clip_len"])
    p.add_argument("--ckpt_dir",        type=str,   default=defaults["ckpt_dir"])
    p.add_argument("--starter_ckpt",    type=str,   default=defaults["starter_ckpt"])
    p.add_argument("--hidden",          type=int,   default=defaults["hidden"])
    p.add_argument("--n_layers",        type=int,   default=defaults["n_layers"])
    p.add_argument("--decoder_refine",  type=int,   default=defaults["decoder_refine"])
    p.add_argument("--freeze_epochs",   type=int,   default=defaults["freeze_epochs"])
    p.add_argument("--lr",              type=float, default=defaults["lr"])
    p.add_argument("--lr_finetune",     type=float, default=defaults["lr_finetune"])
    p.add_argument("--lambda_rate",     type=float, default=defaults["lambda_rate"])
    p.add_argument("--lambda_spec",     type=float, default=defaults["lambda_spec"])
    p.add_argument("--lambda_recon",    type=float, default=defaults["lambda_recon"])
    p.add_argument("--grad_clip",       type=float, default=defaults["grad_clip"])
    p.add_argument("--gain_aug_db",     type=float, default=defaults["gain_aug_db"])
    p.add_argument("--seed",            type=int,   default=defaults["seed"])
    p.add_argument("--log_dir",         type=str,   default=defaults["log_dir"])
    p.add_argument("--csv_path",        type=str,   default=defaults["csv_path"])
    p.add_argument("--dprnn_bn_dim",    type=int,   default=defaults["dprnn_bn_dim"])
    p.add_argument("--dprnn_rnn_hidden", type=int,  default=defaults["dprnn_rnn_hidden"])
    p.add_argument("--epoch_size",      type=int,   default=defaults["epoch_size"])
    p.add_argument("--snr_min_db",      type=float, default=defaults["snr_min_db"])
    p.add_argument("--snr_max_db",      type=float, default=defaults["snr_max_db"])
    aug_group = p.add_mutually_exclusive_group()
    aug_group.add_argument("--no_speed_perturb", dest="speed_perturb", action="store_false")
    aug_group.add_argument("--speed_perturb",    dest="speed_perturb", action="store_true")
    p.set_defaults(speed_perturb=defaults["speed_perturb"])
    aug_group2 = p.add_mutually_exclusive_group()
    aug_group2.add_argument("--no_train_augment", dest="train_augment", action="store_false")
    aug_group2.add_argument("--train_augment",    dest="train_augment", action="store_true")
    p.set_defaults(train_augment=defaults["train_augment"])

    cfg = p.parse_args()
    for k, v in defaults.items():
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
        print("[warn] No GPU — training on CPU will be very slow")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print(f"\n[data] Building dynamic mix dataloaders from {cfg.librimix_root}")
    print(f"       train={cfg.train_split}  val={cfg.val_split}  "
          f"clip_len={cfg.clip_len} ({cfg.clip_len/16000:.1f}s)")
    print(f"[data] epoch_size={cfg.epoch_size}  speed_perturb={cfg.speed_perturb}  "
          f"snr_range=[{cfg.snr_min_db}, {cfg.snr_max_db}] dB")
    print(f"[aug]  dataset augment={'ON' if cfg.train_augment else 'OFF'}  "
          f"per-source gain_aug_db=+-{cfg.gain_aug_db} dB")

    train_loader, val_loader = build_dynamic_mix_dataloaders(
        librimix_root  = cfg.librimix_root,
        train_split    = cfg.train_split,
        val_split      = cfg.val_split,
        clip_len       = cfg.clip_len,
        batch_size     = cfg.batch_size,
        num_workers    = cfg.num_workers,
        epoch_size     = cfg.epoch_size,
        speed_perturb  = cfg.speed_perturb,
        snr_range_db   = (cfg.snr_min_db, cfg.snr_max_db),
        train_augment  = cfg.train_augment,
    )
    print(f"[data] train={len(train_loader.dataset)} items/epoch ({len(train_loader)} batches/ep)  "
          f"val={len(val_loader.dataset)} files ({len(val_loader)} batches/ep)")

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_dir    = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt   = ckpt_dir / "best_2spk.pt"
    latest_ckpt = ckpt_dir / "best_2spk_latest.pt"
    _peek_ckpt  = latest_ckpt if latest_ckpt.exists() else best_ckpt

    # ── Architecture peek for resume ──────────────────────────────────────────
    if cfg.resume and _peek_ckpt.exists():
        _early = torch.load(_peek_ckpt, map_location="cpu", weights_only=False)
        _saved = _early.get("model_cfg", {})
        for k in ("n_filters", "kernel_sz", "stride", "hidden", "n_layers",
                  "snn_chunk", "decoder_refine", "decoder_groups",
                  "dprnn_bn_dim", "dprnn_rnn_hidden"):
            if k in _saved:
                setattr(cfg, k, _saved[k])
        print(f"[resume] Architecture: hidden={cfg.hidden}  n_layers={cfg.n_layers}  "
              f"dropout={cfg.dropout}  snn_mode=dprnn  "
              f"bn_dim={cfg.dprnn_bn_dim}  rnn_hidden={cfg.dprnn_rnn_hidden}")
        del _early, _saved

    # ── Build model ───────────────────────────────────────────────────────────
    model = SNNTasNet(
        n_filters        = cfg.n_filters,
        kernel_sz        = cfg.kernel_sz,
        stride           = cfg.stride,
        hidden           = cfg.hidden,
        n_layers         = cfg.n_layers,
        n_speakers       = cfg.n_speakers,
        dropout          = cfg.dropout,
        snn_mode         = "dprnn",
        snn_chunk        = cfg.snn_chunk,
        decoder_refine   = cfg.decoder_refine,
        decoder_groups   = cfg.decoder_groups,
        use_weight_norm  = False,
        dprnn_bn_dim     = cfg.dprnn_bn_dim,
        dprnn_rnn_hidden = cfg.dprnn_rnn_hidden,
    ).to(device)

    n_params   = sum(p.numel() for p in model.parameters())
    sep_params = sum(p.numel() for p in model.separator.parameters())
    print(f"\n[model] SNNTasNet v15 (dynamic mix, v13-sized DPRNN)  params={n_params:,}  "
          f"separator={sep_params:,}")
    print(f"        bn_dim={cfg.dprnn_bn_dim}  rnn_hidden={cfg.dprnn_rnn_hidden}  "
          f"n_blocks={cfg.n_layers}  chunk={cfg.snn_chunk}")

    if device.type == "cuda":
        print(f"[vram]  {torch.cuda.max_memory_allocated()/1e9:.2f} GB after model init")

    # ── Optimizer: two param groups with freeze ──────────────────────────────
    sep_params_list = list(model.separator.parameters())
    sep_ids = {id(p) for p in sep_params_list}
    enc_dec_params = [p for p in model.parameters() if id(p) not in sep_ids]

    if cfg.freeze_epochs > 0:
        for p in enc_dec_params:
            p.requires_grad_(False)
        optimizer = AdamW(sep_params_list, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print(f"[freeze] Encoder+decoder frozen for {cfg.freeze_epochs} epochs  "
              f"(separator lr={cfg.lr})")
    else:
        optimizer = AdamW([
            {"params": sep_params_list, "lr": cfg.lr},
            {"params": enc_dec_params,  "lr": cfg.lr_finetune},
        ], weight_decay=cfg.weight_decay)

    _T = max(1, cfg.n_epochs - cfg.freeze_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_T, eta_min=1e-6)

    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch       = 1
    best_val          = -1e9
    _actually_resumed = False
    _ckpt             = None
    if cfg.resume and _peek_ckpt.exists():
        print(f"\n[resume] Loading {_peek_ckpt}")
        _ckpt = torch.load(_peek_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model_state"])
        start_epoch       = _ckpt.get("epoch", 0) + 1
        best_val          = _ckpt.get("best_val_si_sdri", _ckpt.get("val_si_sdri", -1e9))
        _actually_resumed = True

        if start_epoch <= cfg.freeze_epochs:
            for p in enc_dec_params:
                p.requires_grad_(False)
            optimizer = AdamW(sep_params_list, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            for p in enc_dec_params:
                p.requires_grad_(True)
            optimizer = AdamW([
                {"params": sep_params_list, "lr": cfg.lr},
                {"params": enc_dec_params,  "lr": cfg.lr_finetune},
            ], weight_decay=cfg.weight_decay)

        _T = max(1, cfg.n_epochs - cfg.freeze_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=_T, eta_min=1e-6)

        if "optimizer_state" in _ckpt:
            saved_groups = _ckpt["optimizer_state"].get("param_groups", [])
            if len(saved_groups) == len(optimizer.param_groups):
                optimizer.load_state_dict(_ckpt["optimizer_state"])
                if "scheduler_state" in _ckpt:
                    scheduler.load_state_dict(_ckpt["scheduler_state"])
            else:
                print(f"[resume] Optimizer groups changed ({len(saved_groups)}→"
                      f"{len(optimizer.param_groups)}), resetting optimizer+scheduler")

        print(f"[resume] Epoch {start_epoch}  best={best_val:+.2f} dB")

    # ── Warmstart: encoder+decoder from v13, fresh separator ─────────────────
    if cfg.warmstart and not _actually_resumed:
        ws_path = Path(cfg.warmstart)
        if ws_path.exists():
            print(f"\n[warmstart] Loading encoder+decoder from {ws_path}")
            ws_ckpt = torch.load(ws_path, map_location=device, weights_only=False)
            if "ema_state" in ws_ckpt and ws_ckpt["ema_state"].get("shadow"):
                shadow = ws_ckpt["ema_state"]["shadow"]
                loaded, skipped = 0, 0
                for name, param in model.named_parameters():
                    if name.startswith("separator."):
                        skipped += 1
                        continue
                    if name in shadow:
                        param.data.copy_(shadow[name])
                        loaded += 1
                print(f"[warmstart] Loaded {loaded} encoder+decoder params from EMA shadow")
                print(f"[warmstart] Skipped {skipped} separator params (fresh DPRNN)")
            else:
                src_state = ws_ckpt.get("model_state", {})
                loaded, skipped = 0, 0
                for name, param in model.named_parameters():
                    if name.startswith("separator."):
                        skipped += 1
                        continue
                    if name in src_state:
                        param.data.copy_(src_state[name])
                        loaded += 1
                print(f"[warmstart] Loaded {loaded} encoder+decoder from model_state")
                print(f"[warmstart] Skipped {skipped} separator params (fresh DPRNN)")
            ws_epoch = ws_ckpt.get("epoch", "?")
            ws_val   = ws_ckpt.get("val_si_sdri", float("nan"))
            print(f"[warmstart] Source: epoch={ws_epoch}  val_SI-SDRi={ws_val:+.2f} dB")
            del ws_ckpt
        else:
            print(f"\n[warn] Warmstart checkpoint not found: {ws_path}. "
                  "Training from scratch.")

    # ── EMA (init after weights are loaded) ───────────────────────────────────
    ema = EMA(model, decay=cfg.ema_decay)
    if _actually_resumed and _ckpt is not None and "ema_state" in _ckpt:
        ema.load_state_dict(_ckpt["ema_state"])

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = Path(cfg.csv_path)
    _header  = ["epoch", "trn_loss", "trn_si_sdri", "val_si_sdri",
                 "gap", "spike_rate", "recon_loss", "spec_loss",
                 "lr", "epoch_time_s", "is_best"]
    if not cfg.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_header)

    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    job_start   = time.time()
    epoch_times: deque = deque(maxlen=10)
    vl: dict = {}

    sep_line = "-" * 105
    print(f"\n{sep_line}")
    print(f"{'Ep':>5}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'Spike':>7}  {'LR':>8}  {'t':>6}  {'ETA':>8}")
    print(sep_line)

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()

        # ── Unfreeze encoder+decoder after freeze phase ───────────────────────
        if epoch == cfg.freeze_epochs + 1 and cfg.freeze_epochs > 0:
            print(f"\n[unfreeze] Epoch {epoch}: unfreezing encoder+decoder "
                  f"(lr_finetune={cfg.lr_finetune})")
            for p in enc_dec_params:
                p.requires_grad_(True)
            optimizer = AdamW([
                {"params": sep_params_list, "lr": cfg.lr},
                {"params": enc_dec_params,  "lr": cfg.lr_finetune},
            ], weight_decay=cfg.weight_decay)
            _T = max(1, cfg.n_epochs - cfg.freeze_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=_T, eta_min=1e-6)

        # ── Train ─────────────────────────────────────────────────────────────
        tr = run_epoch(model, train_loader, optimizer, device, scaler,
                       cfg, is_train=True, ema=ema)

        # ── Validate ──────────────────────────────────────────────────────────
        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.n_epochs)
        if do_val:
            ema.apply_shadow()
            vl = run_epoch(model, val_loader, None, device, None,
                           cfg, is_train=False, ema=None)
            ema.restore()

        # ── Scheduler ─────────────────────────────────────────────────────────
        if epoch > cfg.freeze_epochs:
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

        print(f"{epoch:>5}/{cfg.n_epochs}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{val_str}  {lr:.2e}  {dt:>5.0f}s  {eta:>8}")

        if epoch == 1 and device.type == "cuda":
            print(f"  [vram] peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        # ── TensorBoard ───────────────────────────────────────────────────────
        writer.add_scalar("si_sdri/train",  tr["si_sdri"],  epoch)
        writer.add_scalar("loss/train",     tr["loss"],     epoch)
        writer.add_scalar("lr",             lr,             epoch)
        if do_val:
            writer.add_scalar("si_sdri/val",    vl["si_sdri"],    epoch)
            writer.add_scalar("gap/train_val",  gap,              epoch)
            writer.add_scalar("recon_loss/val", vl["recon_loss"], epoch)

        # ── Checkpoint: best_2spk.pt ──────────────────────────────────────────
        is_best = do_val and vl["si_sdri"] > best_val
        if is_best:
            best_val = vl["si_sdri"]
            _save_ckpt(best_ckpt, epoch, model, ema, optimizer, scheduler,
                       best_val, best_val, cfg)
            print(f"  New best  val={best_val:+.2f} dB  gap={gap:+.1f} dB")

        # ── Checkpoint: latest (every epoch for SLURM resume) ────────────────
        _save_ckpt(latest_ckpt, epoch, model, ema, optimizer, scheduler,
                   vl.get("si_sdri", float("nan")), best_val, cfg)

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(tr["loss"],        4), round(tr["si_sdri"],  4),
                round(vl["si_sdri"],     4) if do_val else "",
                round(gap,               4) if do_val else "",
                round(vl.get("spike_rate", 0.0), 4) if do_val else "",
                round(vl.get("recon_loss",  0.0), 4) if do_val else "",
                round(vl.get("spec_loss",   0.0), 4) if do_val else "",
                lr, round(dt, 1), int(is_best),
            ])

        # ── Wall-time guard ───────────────────────────────────────────────────
        if cfg.max_wall_hours > 0:
            elapsed = time.time() - job_start
            if elapsed >= cfg.max_wall_hours * 3600:
                print(f"\n[wall-time] {elapsed/3600:.2f}h elapsed — "
                      f"checkpoint at {latest_ckpt}")
                print(f"[wall-time] Resubmit with --resume to continue "
                      f"from epoch {epoch + 1}.")
                writer.close()
                return

        if epoch == 3 and vl.get("wform_mean", 1.0) < 1e-4:
            print("\n  WARNING: waveform_mean ~0 after 3 epochs. "
                  "Check --librimix_root path and data integrity.\n")

    print(f"\n{'─'*60}")
    print(f"[done] Training complete.  Best val SI-SDRi = {best_val:+.2f} dB")
    print(f"       Best checkpoint : {best_ckpt}")
    print(f"       CSV log         : {csv_path}")
    writer.close()


if __name__ == "__main__":
    main()
