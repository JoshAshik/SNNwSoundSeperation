"""
oracle_mask_diagnostic.py — Front-end ceiling estimate via oracle STFT masks.

Purpose (v17 gate): before spending A100 time on a finer encoder, bound how much
SI-SDRi each encoder time/frequency resolution can possibly reach. We use STFT
oracle masks (ideal ratio / ideal binary) as a proxy for what a *masking*
separator could achieve at each (window, hop) — mirroring the learned encoder's
analysis resolution:

    encoder k=32, s=16  ->  STFT n_fft=32,  hop=16   (current v13–v16 front-end)
    encoder k=16, s=8   ->  STFT n_fft=16,  hop=8    (proposed v17 Experiment A)

If the finer window's oracle ceiling is NOT meaningfully higher than the coarse
one, the encoder stride is not the bottleneck and Experiment A should be skipped.
Larger reference windows are included for context (absolute oracle ceiling).

Masks use the mixture's phase (standard mixture-phase oracle). Per-speaker
SI-SDRi is reported separately to surface channel collapse asymmetry.

This script does NOT load or train any model — it is pure signal processing.

Usage:
    python3 oracle_mask_diagnostic.py \\
        --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \\
        --split dev --max_samples 200

    # Compare only the two encoder-analog windows on the full dev set:
    python3 oracle_mask_diagnostic.py \\
        --librimix_root /path/to/Libri2Mix/wav16k/max --split dev \\
        --windows 32:16 16:8
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import csv
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16_000

# (n_fft, hop) pairs. First two are the encoder analogs; the rest give context.
DEFAULT_WINDOWS = [(32, 16), (16, 8), (256, 64), (512, 128)]


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR / SI-SDRi
# ─────────────────────────────────────────────────────────────────────────────

def _si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    eps = 1e-8
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = (est * ref).sum()
    ref2 = (ref * ref).sum().clamp(min=eps)
    proj = (dot / ref2) * ref
    noise = est - proj
    ratio = proj.pow(2).sum() / noise.pow(2).sum().clamp(min=eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps)).item()


def _si_sdri(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> float:
    return _si_sdr(est, ref) - _si_sdr(mix, ref)


# ─────────────────────────────────────────────────────────────────────────────
# Oracle mask separation
# ─────────────────────────────────────────────────────────────────────────────

def _stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                      return_complex=True, center=True)


def _istft(X: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=X.device)
    return torch.istft(X, n_fft=n_fft, hop_length=hop, window=win,
                       center=True, length=length)


def oracle_separate(mix: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                    n_fft: int, hop: int, mask_type: str) -> tuple:
    """
    Return (est1, est2) reconstructed with an oracle mask applied to the mixture
    STFT (mixture phase retained). mask_type in {"irm", "ibm"}.
    """
    T = mix.shape[-1]
    M  = _stft(mix, n_fft, hop)
    S1 = _stft(s1,  n_fft, hop)
    S2 = _stft(s2,  n_fft, hop)

    a1 = S1.abs()
    a2 = S2.abs()
    eps = 1e-8

    if mask_type == "irm":
        # Power-ratio mask
        p1, p2 = a1.pow(2), a2.pow(2)
        denom  = (p1 + p2).clamp(min=eps)
        m1, m2 = p1 / denom, p2 / denom
    elif mask_type == "ibm":
        m1 = (a1 >= a2).float()
        m2 = 1.0 - m1
    else:
        raise ValueError(f"unknown mask_type {mask_type}")

    est1 = _istft(M * m1, n_fft, hop, T)
    est2 = _istft(M * m2, n_fft, hop, T)
    return est1, est2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_windows(items):
    out = []
    for it in items:
        a, b = it.split(":")
        out.append((int(a), int(b)))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Oracle STFT-mask SI-SDRi ceiling per front-end resolution")
    p.add_argument("--librimix_root", required=True,
                   help="Libri2Mix wav16k/max root")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train-100"])
    p.add_argument("--max_samples", type=int, default=200,
                   help="Limit to first N samples (0 = all). dev/test = 3000 each.")
    p.add_argument("--windows", nargs="+", default=None,
                   help='Window:hop pairs, e.g. "32:16 16:8". Default: 32:16 16:8 256:64 512:128')
    p.add_argument("--masks", nargs="+", default=["irm", "ibm"],
                   choices=["irm", "ibm"])
    p.add_argument("--output_csv", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    windows = _parse_windows(args.windows) if args.windows else DEFAULT_WINDOWS

    split_dir = Path(args.librimix_root) / args.split
    mix_dir = split_dir / "mix_clean"
    s1_dir  = split_dir / "s1"
    s2_dir  = split_dir / "s2"
    for d in (mix_dir, s1_dir, s2_dir):
        if not d.is_dir():
            print(f"[error] Directory not found: {d}")
            sys.exit(1)

    stems = sorted(pp.stem for pp in mix_dir.glob("*.wav"))
    if args.max_samples > 0:
        stems = stems[:args.max_samples]

    print(f"\n[oracle] split={args.split}  samples={len(stems)}")
    print(f"[oracle] windows (n_fft:hop) = "
          + "  ".join(f"{n}:{h}" for n, h in windows))
    print(f"[oracle] masks = {args.masks}")
    print(f"[oracle] mixture-phase oracle; SI-SDRi vs mixture\n")

    # accum[(n_fft, hop, mask)] = [sum_sdri, sum_sdri_s1, sum_sdri_s2, count]
    accum = {(n, h, m): [0.0, 0.0, 0.0, 0]
             for (n, h) in windows for m in args.masks}

    for i, stem in enumerate(stems):
        mix, sr = torchaudio.load(str(mix_dir / f"{stem}.wav"))
        s1, _   = torchaudio.load(str(s1_dir / f"{stem}.wav"))
        s2, _   = torchaudio.load(str(s2_dir / f"{stem}.wav"))
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")

        mix = mix.mean(0).to(device)
        s1  = s1.mean(0).to(device)
        s2  = s2.mean(0).to(device)
        T = min(mix.shape[-1], s1.shape[-1], s2.shape[-1])
        mix, s1, s2 = mix[:T], s1[:T], s2[:T]

        for (n_fft, hop) in windows:
            for m in args.masks:
                est1, est2 = oracle_separate(mix, s1, s2, n_fft, hop, m)
                L = min(est1.shape[-1], T)
                d1 = _si_sdri(est1[:L], s1[:L], mix[:L])
                d2 = _si_sdri(est2[:L], s2[:L], mix[:L])
                a = accum[(n_fft, hop, m)]
                a[0] += (d1 + d2) / 2
                a[1] += d1
                a[2] += d2
                a[3] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(stems):
            print(f"  [{i+1:>5}/{len(stems)}] processed", flush=True)

    print("\n" + "─" * 78)
    print(f"{'window':>10}  {'mask':>4}  {'SI-SDRi':>9}  "
          f"{'s1':>8}  {'s2':>8}  {'|s1-s2|':>8}")
    print("─" * 78)
    rows = []
    for (n_fft, hop) in windows:
        for m in args.masks:
            s_all, s_s1, s_s2, c = accum[(n_fft, hop, m)]
            mean_all = s_all / max(c, 1)
            mean_s1  = s_s1 / max(c, 1)
            mean_s2  = s_s2 / max(c, 1)
            asym     = abs(mean_s1 - mean_s2)
            wlabel   = f"{n_fft}:{hop}"
            print(f"{wlabel:>10}  {m:>4}  {mean_all:>+8.2f}  "
                  f"{mean_s1:>+8.2f}  {mean_s2:>+8.2f}  {asym:>8.2f}")
            rows.append((wlabel, m, mean_all, mean_s1, mean_s2, asym))
    print("─" * 78)

    # Headline comparison: the two encoder-analog windows.
    enc_pairs = [(n, h) for (n, h) in windows if (n, h) in [(32, 16), (16, 8)]]
    if (32, 16) in [(n, h) for n, h in windows] and (16, 8) in [(n, h) for n, h in windows]:
        for m in args.masks:
            coarse = accum[(32, 16, m)]
            fine   = accum[(16, 8, m)]
            c_mean = coarse[0] / max(coarse[3], 1)
            f_mean = fine[0] / max(fine[3], 1)
            print(f"[verdict:{m}] stride-8 vs stride-16 oracle gain = "
                  f"{f_mean - c_mean:+.2f} dB  "
                  f"(32:16 {c_mean:+.2f} -> 16:8 {f_mean:+.2f})")
        print("[verdict] If the gain is small (< ~1 dB), the encoder stride is "
              "NOT the ceiling — reconsider Experiment A before training.")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window", "mask", "si_sdri", "si_sdri_s1",
                        "si_sdri_s2", "asym"])
            for r in rows:
                w.writerow([r[0], r[1], round(r[2], 4), round(r[3], 4),
                            round(r[4], 4), round(r[5], 4)])
        print(f"\n[csv] Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
