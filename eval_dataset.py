"""
eval_dataset.py — Batch SI-SDRi evaluation on Libri2Mix splits.

Evaluates a trained checkpoint on dev or test splits and reports per-sample
SI-SDRi statistics. Supports a --max_samples flag for fast dev50 iteration.

Usage:
    # Full dev evaluation:
    python3 eval_dataset.py --model checkpoints_v11/best_2spk.pt \\
        --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \\
        --split dev

    # Fast dev50 subset:
    python3 eval_dataset.py --model checkpoints_v11/best_2spk.pt \\
        --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \\
        --split dev --max_samples 50

    # Full test + CSV output:
    python3 eval_dataset.py --model checkpoints_v12/best_2spk.pt \\
        --librimix_root /path/to/Libri2Mix/wav16k/max \\
        --split test --output_csv eval_test_v12.csv

Outputs:
    - Per-sample SI-SDRi printed to stdout
    - Summary: mean, median, std, worst-5
    - Optional CSV with per-sample results
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

try:
    from sep_model_v10 import SNNTasNet
except ImportError:
    from sep_model import SNNTasNet

SAMPLE_RATE = 16_000


def load_model(ckpt_path: str, device: torch.device) -> SNNTasNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = ckpt.get("model_cfg", {})
    defaults = dict(n_filters=256, kernel_sz=32, stride=16,
                    hidden=512, n_layers=6, dropout=0.35,
                    snn_mode="gru_stateful", snn_chunk=100,
                    use_weight_norm=False,
                    decoder_refine=3, decoder_groups=8)
    defaults.update(model_cfg)

    n_speakers = defaults.pop("n_speakers", 2)
    model = SNNTasNet(**defaults, n_speakers=n_speakers).to(device)

    if "ema_state" in ckpt and ckpt["ema_state"].get("shadow"):
        shadow = ckpt["ema_state"]["shadow"]
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
        weight_src = "EMA shadow"
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)
        weight_src = "raw model_state"

    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_si = ckpt.get("val_si_sdri", float("nan"))
    print(f"[model] {Path(ckpt_path).name}  epoch={epoch}  "
          f"val_SI-SDRi={val_si:+.2f} dB  weights={weight_src}")
    return model


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


def si_sdri(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> float:
    return _si_sdr(est.flatten(), ref.flatten()) - _si_sdr(mix.flatten(), ref.flatten())


@torch.no_grad()
def evaluate_sample(model, mix: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                    device: torch.device) -> float:
    mix_in = mix.unsqueeze(0).to(device)
    out = model(mix_in)
    est = out["separated"]

    T = mix.shape[-1]
    T_out = est.shape[-1]
    if T_out > T:
        est = est[..., :T]
    elif T_out < T:
        est = F.pad(est, (0, T - T_out))

    est_0 = est[0, 0].cpu()
    est_1 = est[0, 1].cpu()
    mix_flat = mix.flatten()
    s1_flat = s1.flatten()
    s2_flat = s2.flatten()

    sdri_p0 = (si_sdri(est_0, s1_flat, mix_flat) + si_sdri(est_1, s2_flat, mix_flat)) / 2
    sdri_p1 = (si_sdri(est_0, s2_flat, mix_flat) + si_sdri(est_1, s1_flat, mix_flat)) / 2
    return max(sdri_p0, sdri_p1)


def main():
    p = argparse.ArgumentParser(description="Batch SI-SDRi evaluation on Libri2Mix")
    p.add_argument("--model", required=True, help="Checkpoint path")
    p.add_argument("--librimix_root", required=True, help="Libri2Mix wav16k/max root")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train-100"],
                   help="Which split to evaluate")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit to first N samples (0 = all)")
    p.add_argument("--output_csv", type=str, default="",
                   help="Save per-sample results to this CSV")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = load_model(args.model, device)

    split_dir = Path(args.librimix_root) / args.split
    mix_dir = split_dir / "mix_clean"
    s1_dir = split_dir / "s1"
    s2_dir = split_dir / "s2"

    for d in (mix_dir, s1_dir, s2_dir):
        if not d.is_dir():
            print(f"[error] Directory not found: {d}")
            sys.exit(1)

    stems = sorted(p.stem for p in mix_dir.glob("*.wav"))
    if args.max_samples > 0:
        stems = stems[:args.max_samples]

    print(f"\n[eval] split={args.split}  samples={len(stems)}")
    print("-" * 70)

    results = []
    for i, stem in enumerate(stems):
        mix_wav, sr = torchaudio.load(str(mix_dir / f"{stem}.wav"))
        s1_wav, _   = torchaudio.load(str(s1_dir / f"{stem}.wav"))
        s2_wav, _   = torchaudio.load(str(s2_dir / f"{stem}.wav"))

        mix_wav = mix_wav.squeeze(0)
        s1_wav = s1_wav.squeeze(0)
        s2_wav = s2_wav.squeeze(0)

        T = min(mix_wav.shape[-1], s1_wav.shape[-1], s2_wav.shape[-1])
        mix_wav = mix_wav[:T]
        s1_wav = s1_wav[:T]
        s2_wav = s2_wav[:T]

        sdri = evaluate_sample(model, mix_wav, s1_wav, s2_wav, device)
        results.append((stem, sdri))

        if (i + 1) % 50 == 0 or (i + 1) == len(stems):
            running_mean = sum(r[1] for r in results) / len(results)
            print(f"  [{i+1:>5}/{len(stems)}]  "
                  f"this={sdri:>+.2f} dB  running_mean={running_mean:>+.2f} dB")

    print("-" * 70)
    scores = [r[1] for r in results]
    scores_t = torch.tensor(scores)

    mean_sdri = scores_t.mean().item()
    median_sdri = scores_t.median().item()
    std_sdri = scores_t.std().item()

    print(f"\n[summary] {args.split} ({len(scores)} samples)")
    print(f"  Mean SI-SDRi:   {mean_sdri:>+.2f} dB")
    print(f"  Median SI-SDRi: {median_sdri:>+.2f} dB")
    print(f"  Std:            {std_sdri:.2f} dB")
    print(f"  Min:            {scores_t.min().item():>+.2f} dB")
    print(f"  Max:            {scores_t.max().item():>+.2f} dB")

    worst = sorted(results, key=lambda x: x[1])[:5]
    print(f"\n  Worst 5:")
    for stem, s in worst:
        print(f"    {stem}  {s:>+.2f} dB")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stem", "si_sdri_db"])
            for stem, s in results:
                w.writerow([stem, round(s, 4)])
        print(f"\n[csv] Saved {len(results)} results to {csv_path}")


if __name__ == "__main__":
    main()
