"""
two_speaker_train_v19.py — v17 recipe on train-360, WARMSTART recovery + hardening.

v19's first attempt (v17 recipe run from scratch on train-360, slurm_v19) climbed
cleanly to val +7.95 dB (ep26) then DIVERGED: val collapsed +7.95 -> -12.8 (ep32)
-> NaN (ep36). Root cause (from v19_train_log.csv): the plateau scheduler never
decayed the LR because val kept setting new bests, so the model ran ~26 epochs x
6350 batches = ~165k gradient steps at the FULL lr=1e-3. Adam at a high LR for that
many steps on the larger dataset walked off a cliff (a gradual collapse ep27-35,
then fp16 overflow -> NaN). The LR halving at ep32 was the scheduler REACTING to the
crash, not the trigger.

This script recovers and continues, without redoing the 26 good epochs:

  1. WARMSTART the full model from the +7.95 checkpoint (checkpoints_v19/best_2spk.pt,
     EMA shadow — the exact weights that scored +7.95), fresh optimizer + scheduler.
  2. lr = 3e-4 (PRIMARY FIX) — well below the 1e-3 that diverged, in the stable
     fine-tuning regime; the plateau schedule decays from there.
  3. Loss computed in fp32 (DEFENSIVE) — SI-SDR sum-of-squares over 64000 samples can
     overflow fp16; this is what turned the divergence into NaN.
  4. Skip-batch guard (DEFENSIVE) — any non-finite loss is dropped before backward(),
     so a single pathological batch (e.g. a near-silent "max"-mode zero-padded target)
     can never corrupt the weights again.

Writes to a FRESH ckpt_dir (checkpoints_v19b) so the NaN latest from the first attempt
can't be resumed. two_speaker_train_v17.py is left untouched (still the +9.35 best on
train-100). Everything else matches the v17 recipe: finer encoder k=16/s=8, DPRNN
bn_dim=64/rnn_hidden=128, pure SI-SDR + 0.1*spec, augmentation off.

Resume:    --resume loads {ckpt_dir}/best_2spk_latest.pt if present (SLURM self-resubmit);
           otherwise it warmstarts from --warmstart.
SLURM:     slurm_v19_twospeaker.sh (self-resubmitting, ~6 h wall).

Usage (smoke test):
    python3 two_speaker_train_v19.py --n_epochs 2 --batch_size 2 --num_workers 0 \\
        --warmstart "" --librimix_root ./data/librimix --max_wall_hours 0

Usage (ARC):
    sbatch slurm_v19_twospeaker.sh
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sep_model_v10     import SNNTasNet
from librimix_dataset  import build_librimix_dataloaders, DEFAULT_CLIP_LEN


# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (v17: finer encoder, from scratch, pure SI-SDR, fixed real mixtures)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # ── dataset ──────────────────────────────────────────────────────────────
    librimix_root     = "/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max",
    train_split       = "train-360",   # v19: the DATA lever (3.6x real mixtures)
    val_split         = "dev",
    clip_len          = DEFAULT_CLIP_LEN,
    train_augment     = False,     # C: augmentation OFF (we are underfitting)
    # ── model (Experiment A: finer front-end) ─────────────────────────────────
    n_filters         = 256,
    kernel_sz         = 16,        # A: was 32
    stride            = 8,         # A: was 16  (-> ~2x frames, ~8000 for 4 s)
    hidden            = 128,
    n_layers          = 6,
    n_speakers        = 2,
    dropout           = 0.1,
    snn_chunk         = 200,       # ~sqrt(2*frames); keep ~200 at stride 8
    decoder_refine    = 3,
    decoder_groups    = 8,
    dprnn_bn_dim      = 64,        # v13 size — capacity (F) held for later
    dprnn_rnn_hidden  = 128,
    # ── training (v19: warmstart recovery at a STABLE lr) ──────────────────────
    n_epochs          = 90,
    batch_size        = 8,         # held at v17's value (clean data comparison)
    num_workers       = 8,
    lr                = 3e-4,      # PRIMARY FIX: below the 1e-3 that diverged
    weight_decay      = 0.0,
    plateau_factor    = 0.5,       # halve on plateau
    plateau_patience  = 3,         # in val-checks (val_every epochs each)
    plateau_threshold = 1e-2,      # ABS dB; rel mode misbehaves near 0/neg SI-SDRi
    min_lr            = 1e-6,
    val_every         = 2,         # longer epochs on train-360 → validate more often
    grad_clip         = 5.0,
    ema_decay         = 0.999,
    seed              = 42,
    gain_aug_db       = 0.0,       # per-source gain aug OFF
    # ── loss (pure SI-SDR, computed in fp32 — see pit_forward) ─────────────────
    lambda_spec       = 0.1,
    lambda_rate       = 0.0,
    lambda_recon      = 0.0,
    # ── warmstart / paths ──────────────────────────────────────────────────────
    warmstart         = "./checkpoints_v19/best_2spk.pt",  # the +7.95 weights
    ckpt_dir          = "./checkpoints_v19b",              # FRESH — avoids NaN latest
    log_dir           = "./runs_v19b",
    csv_path          = "./v19b_train_log.csv",
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
# Per-source gain augmentation  (training only; default OFF in v17)
# ─────────────────────────────────────────────────────────────────────────────

def apply_gain_aug(t1: torch.Tensor, t2: torch.Tensor, gain_db: float) -> tuple:
    B = t1.shape[0]
    g1 = 10.0 ** (t1.new_empty(B, 1).uniform_(-gain_db, gain_db) / 20.0)
    g2 = 10.0 ** (t2.new_empty(B, 1).uniform_(-gain_db, gain_db) / 20.0)
    t1_aug  = t1 * g1
    t2_aug  = t2 * g2
    return t1_aug + t2_aug, t1_aug, t2_aug


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


_HANN_CACHE: dict = {}


def _hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, str(device))
    win = _HANN_CACHE.get(key)
    if win is None:
        win = torch.hann_window(n_fft, device=device)
        _HANN_CACHE[key] = win
    return win


def _mr_spectral_loss(est: torch.Tensor, ref: torch.Tensor,
                      fft_sizes: tuple = (256, 512, 1024)) -> torch.Tensor:
    total = est.new_zeros(())
    B, T  = est.shape
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = _hann_window(n_fft, est.device)
        kw  = dict(n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True, center=True)
        S_e = torch.stft(est.reshape(B, T), **kw).abs().clamp(min=1e-8).log()
        S_r = torch.stft(ref.reshape(B, T), **kw).abs().clamp(min=1e-8).log()
        total = total + F.l1_loss(S_e, S_r)
    return total / len(fft_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# PIT forward pass  (pure SI-SDR + small spectral term; recon/rate weighted 0)
# ─────────────────────────────────────────────────────────────────────────────

def pit_forward(model: SNNTasNet, mixture: torch.Tensor,
                t1: torch.Tensor, t2: torch.Tensor, cfg) -> tuple:
    B, T = mixture.shape
    out  = model(mixture)                     # forward under caller's autocast (fp16)
    est  = out["separated"]

    T_out = est.shape[-1]
    if T_out > T:
        est = est[..., :T]
    elif T_out < T:
        est = F.pad(est, (0, T - T_out))

    # HARDENING: compute the entire loss in fp32 with autocast disabled. The v19
    # divergence turned into NaN because the SI-SDR sum-of-squares over 64000
    # samples overflows fp16; fp32 has the range to hold it. est is cast up from
    # the fp16 forward; gradients still flow back to the fp16/fp32 model.
    with autocast("cuda", enabled=False):
        est = est.float()
        t1f, t2f, mixf = t1.float(), t2.float(), mixture.float()

        sdr_p0 = (_si_sdr(est[:, 0], t1f) + _si_sdr(est[:, 1], t2f)) / 2
        sdr_p1 = (_si_sdr(est[:, 0], t2f) + _si_sdr(est[:, 1], t1f)) / 2

        neg_sdr_stack       = torch.stack([-sdr_p0, -sdr_p1], dim=1)
        best_neg, best_perm = neg_sdr_stack.min(dim=1)
        si_sdr_loss         = best_neg.mean()

        total = si_sdr_loss
        spec_val = est.new_zeros(())
        if cfg.lambda_spec > 0.0:
            idx_t1  = best_perm.view(B, 1, 1).expand(B, 1, T).long()
            idx_t2  = (1 - best_perm).view(B, 1, 1).expand(B, 1, T).long()
            est_t1  = est.gather(1, idx_t1).squeeze(1)
            est_t2  = est.gather(1, idx_t2).squeeze(1)
            spec_loss = (_mr_spectral_loss(est_t1, t1f) + _mr_spectral_loss(est_t2, t2f)) / 2
            total = total + cfg.lambda_spec * spec_loss
            spec_val = spec_loss.detach()

        with torch.no_grad():
            best_sdr    = -best_neg
            mix_sdr_avg = (_si_sdr(mixf, t1f) + _si_sdr(mixf, t2f)) / 2
            si_sdri_val = (best_sdr - mix_sdr_avg).mean()

    metrics = {
        "loss":       total.detach(),
        "si_sdri":    si_sdri_val,
        "spec_loss":  spec_val,
        "wform_mean": est.detach().abs().mean(),
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
    n_skipped    = 0
    t_data       = time.time()

    for batch_idx, (mixture, t1, t2) in enumerate(loader):
        mixture = mixture.to(device, non_blocking=True)
        t1      = t1.to(device, non_blocking=True)
        t2      = t2.to(device, non_blocking=True)

        if is_train and cfg.gain_aug_db > 0.0:
            mixture, t1, t2 = apply_gain_aug(t1, t2, cfg.gain_aug_db)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")), \
                torch.set_grad_enabled(is_train):
            loss, metrics = pit_forward(model, mixture, t1, t2, cfg)

        # HARDENING: never let a non-finite loss touch the weights. A single
        # pathological batch (e.g. a near-silent max-mode zero-padded target)
        # is dropped rather than corrupting the model — this is the guard that
        # would have stopped v19's NaN cascade.
        if is_train and not torch.isfinite(loss):
            n_skipped += 1
            continue

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        if is_train and batch_idx % 100 == 0 and batch_idx > 0:
            t_batch = time.time() - t_data
            print(f"  [batch {batch_idx:>4}/{len(loader)}] "
                  f"loss={metrics['loss'].item():.4f}  "
                  f"SI-SDRi={metrics['si_sdri'].item():>+.2f} dB  {t_batch:.1f}s",
                  flush=True)
            t_data = time.time()

    if is_train and n_skipped:
        print(f"  [guard] skipped {n_skipped} non-finite-loss batch(es)", flush=True)

    inv = 1.0 / max(n_batches, 1)
    out = {}
    for k, v in totals.items():
        out[k] = (v * inv).item() if isinstance(v, torch.Tensor) else v * inv
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v19 warmstart recovery on train-360 (lower lr + fp32 loss + NaN guard)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--warmstart",       type=str,   default=defaults["warmstart"],
                   help="checkpoint to warmstart full model from (empty = from scratch)")
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
    p.add_argument("--kernel_sz",       type=int,   default=defaults["kernel_sz"])
    p.add_argument("--stride",          type=int,   default=defaults["stride"])
    p.add_argument("--hidden",          type=int,   default=defaults["hidden"])
    p.add_argument("--n_layers",        type=int,   default=defaults["n_layers"])
    p.add_argument("--snn_chunk",       type=int,   default=defaults["snn_chunk"])
    p.add_argument("--decoder_refine",  type=int,   default=defaults["decoder_refine"])
    p.add_argument("--lr",              type=float, default=defaults["lr"])
    p.add_argument("--weight_decay",    type=float, default=defaults["weight_decay"])
    p.add_argument("--plateau_factor",  type=float, default=defaults["plateau_factor"])
    p.add_argument("--plateau_patience", type=int,  default=defaults["plateau_patience"])
    p.add_argument("--plateau_threshold", type=float, default=defaults["plateau_threshold"])
    p.add_argument("--min_lr",          type=float, default=defaults["min_lr"])
    p.add_argument("--lambda_spec",     type=float, default=defaults["lambda_spec"])
    p.add_argument("--lambda_recon",    type=float, default=defaults["lambda_recon"])
    p.add_argument("--lambda_rate",     type=float, default=defaults["lambda_rate"])
    p.add_argument("--grad_clip",       type=float, default=defaults["grad_clip"])
    p.add_argument("--gain_aug_db",     type=float, default=defaults["gain_aug_db"])
    p.add_argument("--seed",            type=int,   default=defaults["seed"])
    p.add_argument("--log_dir",         type=str,   default=defaults["log_dir"])
    p.add_argument("--csv_path",        type=str,   default=defaults["csv_path"])
    p.add_argument("--dprnn_bn_dim",    type=int,   default=defaults["dprnn_bn_dim"])
    p.add_argument("--dprnn_rnn_hidden", type=int,  default=defaults["dprnn_rnn_hidden"])
    aug_group = p.add_mutually_exclusive_group()
    aug_group.add_argument("--no_train_augment", dest="train_augment", action="store_false")
    aug_group.add_argument("--train_augment",    dest="train_augment", action="store_true")
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

    # ── Dataloaders: fixed REAL Libri2Mix mixtures, augment OFF ───────────────
    print(f"\n[data] Building fixed Libri2Mix dataloaders from {cfg.librimix_root}")
    print(f"       train={cfg.train_split}  val={cfg.val_split}  "
          f"clip_len={cfg.clip_len} ({cfg.clip_len/16000:.1f}s)  "
          f"augment={'ON' if cfg.train_augment else 'OFF'}")
    train_loader, val_loader = build_librimix_dataloaders(
        librimix_root = cfg.librimix_root,
        train_split   = cfg.train_split,
        val_split     = cfg.val_split,
        clip_len      = cfg.clip_len,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
        train_augment = cfg.train_augment,
    )
    print(f"[data] train={len(train_loader.dataset)} files ({len(train_loader)} batches/ep)  "
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
        print(f"[resume] Architecture: kernel_sz={cfg.kernel_sz} stride={cfg.stride}  "
              f"hidden={cfg.hidden}  n_layers={cfg.n_layers}  snn_mode=dprnn  "
              f"bn_dim={cfg.dprnn_bn_dim}  rnn_hidden={cfg.dprnn_rnn_hidden}")
        del _early, _saved

    # ── Build model (from scratch — no warmstart, no freeze) ──────────────────
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
    print(f"\n[model] SNNTasNet v19 (finer encoder k={cfg.kernel_sz}/s={cfg.stride}, "
          f"warmstart recovery)  params={n_params:,}  separator={sep_params:,}")
    print(f"        bn_dim={cfg.dprnn_bn_dim}  rnn_hidden={cfg.dprnn_rnn_hidden}  "
          f"n_blocks={cfg.n_layers}  chunk={cfg.snn_chunk}")
    print(f"[loss]  pure SI-SDR + {cfg.lambda_spec}*spec  "
          f"(lambda_recon={cfg.lambda_recon}, lambda_rate={cfg.lambda_rate})")

    # ── Optimizer + plateau scheduler (fresh, at the lower stable lr) ─────────
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.plateau_factor,
        patience=cfg.plateau_patience, threshold=cfg.plateau_threshold,
        threshold_mode="abs", min_lr=cfg.min_lr)
    print(f"[optim] Adam lr={cfg.lr}  ReduceLROnPlateau(mode=max, "
          f"factor={cfg.plateau_factor}, patience={cfg.plateau_patience} val-checks)  "
          f"grad_clip={cfg.grad_clip}")

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
        start_epoch = _ckpt.get("epoch", 0) + 1
        best_val    = _ckpt.get("best_val_si_sdri", _ckpt.get("val_si_sdri", -1e9))
        if "optimizer_state" in _ckpt:
            optimizer.load_state_dict(_ckpt["optimizer_state"])
        if "scheduler_state" in _ckpt:
            scheduler.load_state_dict(_ckpt["scheduler_state"])
        _actually_resumed = True
        print(f"[resume] Epoch {start_epoch}  best={best_val:+.2f} dB")
    elif cfg.resume:
        print(f"\n[resume] No checkpoint at {_peek_ckpt} — will try warmstart.")

    # ── Warmstart: load the FULL model from the +7.95 checkpoint (EMA shadow) ──
    # Only on a fresh start (no resume checkpoint yet). Optimizer + scheduler stay
    # fresh at the new (lower) lr — we do NOT restore the pre-divergence optimizer
    # state, which carried the Adam moments that walked off the cliff.
    if not _actually_resumed and cfg.warmstart:
        ws_path = Path(cfg.warmstart)
        if ws_path.exists():
            print(f"\n[warmstart] Loading full model from {ws_path}")
            ws = torch.load(ws_path, map_location=device, weights_only=False)
            shadow = ws.get("ema_state", {}).get("shadow")
            if shadow:
                loaded = 0
                for name, param in model.named_parameters():
                    if name in shadow:
                        param.data.copy_(shadow[name]); loaded += 1
                # buffers too (GroupNorm etc. have none trainable, but be safe)
                print(f"[warmstart] Loaded {loaded} params from EMA shadow "
                      f"(source val={ws.get('best_val_si_sdri', float('nan')):+.2f} dB)")
            else:
                model.load_state_dict(ws["model_state"])
                print(f"[warmstart] Loaded model_state "
                      f"(source val={ws.get('best_val_si_sdri', float('nan')):+.2f} dB)")
            del ws
        else:
            print(f"\n[warn] Warmstart checkpoint not found: {ws_path} — from scratch.")

    # ── EMA (init after weights loaded) ───────────────────────────────────────
    ema = EMA(model, decay=cfg.ema_decay)
    if _actually_resumed and _ckpt is not None and "ema_state" in _ckpt:
        ema.load_state_dict(_ckpt["ema_state"])

    if device.type == "cuda":
        print(f"[vram]  {torch.cuda.max_memory_allocated()/1e9:.2f} GB after model init")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = Path(cfg.csv_path)
    _header  = ["epoch", "trn_loss", "trn_si_sdri", "val_si_sdri",
                "gap", "spec_loss", "lr", "epoch_time_s", "is_best"]
    if not cfg.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_header)

    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    job_start   = time.time()
    epoch_times: deque = deque(maxlen=10)
    vl: dict = {}

    sep_line = "-" * 100
    print(f"\n{sep_line}")
    print(f"{'Ep':>5}  {'TrnLoss':>8}  {'TrnSI-SDRi':>10}  "
          f"{'ValSI-SDRi':>10}  {'Gap':>7}  {'LR':>8}  {'t':>6}  {'ETA':>8}")
    print(sep_line)

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.time()

        tr = run_epoch(model, train_loader, optimizer, device, scaler,
                       cfg, is_train=True, ema=ema)

        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.n_epochs)
        if do_val:
            ema.apply_shadow()
            vl = run_epoch(model, val_loader, None, device, None,
                           cfg, is_train=False, ema=None)
            ema.restore()
            # Plateau LR steps on the validation metric (max SI-SDRi).
            scheduler.step(vl["si_sdri"])

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0
        epoch_times.append(dt)
        avg_t  = sum(epoch_times) / len(epoch_times)
        remain = avg_t * (cfg.n_epochs - epoch)
        eta    = f"{int(remain//3600)}h{int((remain%3600)//60):02d}m"
        gap    = tr["si_sdri"] - vl.get("si_sdri", float("nan"))

        if do_val:
            val_str = f"{vl['si_sdri']:>+9.2f}dB  {gap:>+6.1f}dB"
        else:
            val_str = f"{'---':>9}    {'---':>6}  "

        print(f"{epoch:>5}/{cfg.n_epochs}  "
              f"{tr['loss']:>8.4f}  {tr['si_sdri']:>+9.2f}dB  "
              f"{val_str}  {lr:.2e}  {dt:>5.0f}s  {eta:>8}")

        if epoch == start_epoch and device.type == "cuda":
            print(f"  [vram] peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        writer.add_scalar("si_sdri/train", tr["si_sdri"], epoch)
        writer.add_scalar("loss/train",    tr["loss"],    epoch)
        writer.add_scalar("lr",            lr,            epoch)
        if do_val:
            writer.add_scalar("si_sdri/val",   vl["si_sdri"], epoch)
            writer.add_scalar("gap/train_val", gap,           epoch)

        is_best = do_val and vl["si_sdri"] > best_val
        if is_best:
            best_val = vl["si_sdri"]
            _save_ckpt(best_ckpt, epoch, model, ema, optimizer, scheduler,
                       best_val, best_val, cfg)
            print(f"  New best  val={best_val:+.2f} dB  gap={gap:+.1f} dB")

        _save_ckpt(latest_ckpt, epoch, model, ema, optimizer, scheduler,
                   vl.get("si_sdri", float("nan")), best_val, cfg)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(tr["loss"],    4), round(tr["si_sdri"], 4),
                round(vl["si_sdri"], 4) if do_val else "",
                round(gap,           4) if do_val else "",
                round(vl.get("spec_loss", 0.0), 4) if do_val else "",
                lr, round(dt, 1), int(is_best),
            ])

        if cfg.max_wall_hours > 0:
            elapsed = time.time() - job_start
            if elapsed >= cfg.max_wall_hours * 3600:
                print(f"\n[wall-time] {elapsed/3600:.2f}h elapsed — "
                      f"checkpoint at {latest_ckpt}")
                print(f"[wall-time] Resubmit with --resume to continue "
                      f"from epoch {epoch + 1}.")
                writer.close()
                return

        if epoch == start_epoch + 2 and vl.get("wform_mean", 1.0) < 1e-4:
            print("\n  WARNING: waveform_mean ~0 after 3 epochs. "
                  "Check --librimix_root path and data integrity.\n")

    print(f"\n{'─'*60}")
    print(f"[done] Training complete.  Best val SI-SDRi = {best_val:+.2f} dB")
    print(f"       Best checkpoint : {best_ckpt}")
    print(f"       CSV log         : {csv_path}")
    writer.close()


if __name__ == "__main__":
    main()
