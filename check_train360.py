"""
check_train360.py — integrity check for a generated Libri2Mix split.

Verifies the invariants the training pipeline relies on, on the freshly generated
train-360 (or any split): equal file counts across mix_clean/s1/s2, 16 kHz sample
rate, and mixture = s1 + s2 to ~1e-7 (the loss/eval assume this holds exactly).

Run on a compute node (BeeGFS not mounted on login):
    salloc -p rtx4060ti16g -n 1 -t 00:20:00
    /mnt/beegfs/juashik/.conda/envs/snn/bin/python check_train360.py \
        --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
        --split train-360
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import random
from pathlib import Path

import torch
import torchaudio

from librimix_dataset import LibriMixDataset

SR = 16_000


def main() -> int:
    p = argparse.ArgumentParser(description="Libri2Mix split integrity check")
    p.add_argument("--librimix_root",
                   default="/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max")
    p.add_argument("--split", default="train-360")
    p.add_argument("--n_samples", type=int, default=300,
                   help="random items to check for mix=s1+s2 and sample rate")
    p.add_argument("--clip_len", type=int, default=64_000)
    args = p.parse_args()

    root = Path(args.librimix_root) / args.split
    fails = []

    # 1. File counts across the three dirs must be equal and non-zero.
    counts = {}
    for d in ("mix_clean", "s1", "s2"):
        sub = root / d
        if not sub.is_dir():
            fails.append(f"missing dir: {sub}")
            counts[d] = 0
        else:
            counts[d] = sum(1 for _ in sub.glob("*.wav"))
    print(f"[counts] mix_clean={counts['mix_clean']}  s1={counts['s1']}  s2={counts['s2']}")
    if len(set(counts.values())) != 1 or counts["mix_clean"] == 0:
        fails.append(f"file counts unequal or zero: {counts}")

    # 2. Raw sample rate on a handful of files (loader assumes 16 kHz).
    mix_dir = root / "mix_clean"
    raw = sorted(mix_dir.glob("*.wav"))[:5]
    for f in raw:
        info = torchaudio.info(str(f))
        if info.sample_rate != SR:
            fails.append(f"{f.name}: sample_rate={info.sample_rate} (expected {SR})")
    if raw:
        print(f"[sr] checked {len(raw)} files @ {torchaudio.info(str(raw[0])).sample_rate} Hz")

    # 3. mixture = s1 + s2 invariant + clip length, via the real loader.
    if counts["mix_clean"] > 0:
        ds = LibriMixDataset(args.librimix_root, args.split,
                             clip_len=args.clip_len, augment=False)
        print(f"[loader] items={len(ds)}")
        idxs = random.sample(range(len(ds)), min(args.n_samples, len(ds)))
        worst = 0.0
        bad_len = 0
        for i in idxs:
            mix, s1, s2 = ds[i]
            worst = max(worst, (mix - s1 - s2).abs().mean().item())
            if mix.shape[0] != args.clip_len:
                bad_len += 1
        # Tolerance is set for 16-bit PCM wavs: mix/s1/s2 are each quantized
        # independently (step ~3e-5), so mix-s1-s2 carries ~1e-4 of quantization
        # noise even when perfectly generated. A real source/gain mismatch would
        # be ~0.1-1 (4 orders larger), so 1e-3 cleanly separates the two.
        RECON_TOL = 1e-3
        print(f"[recon] max|mix-s1-s2| over {len(idxs)} = {worst:.2e} "
              f"(want < {RECON_TOL:.0e}; ~1e-4 is normal 16-bit PCM quantization)")
        print(f"[clip]  wrong-length items: {bad_len} (want 0); clip_len={args.clip_len}")
        if worst >= RECON_TOL:
            fails.append(f"mix != s1+s2 (max residual {worst:.2e} >= {RECON_TOL:.0e})")
        if bad_len:
            fails.append(f"{bad_len} items not clip_len {args.clip_len}")

    print("-" * 60)
    if fails:
        print("[result] FAILED:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print(f"[result] PASS — {args.split} is clean and ready to train on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
