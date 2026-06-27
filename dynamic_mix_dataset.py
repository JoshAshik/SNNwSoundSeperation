"""
dynamic_mix_dataset.py — On-the-fly clean speech mixing for SNN-TasNet v15+.

Instead of fixed pre-generated Libri2Mix pairs, this dataset:
  1. Pools all clean source utterances from s1/ and s2/ in the train split.
  2. Each __getitem__: randomly samples 2 utterances from the pool.
  3. Applies optional speed perturbation (0.95–1.05x) independently.
  4. Random-crops or tile-pads both to clip_len.
  5. Applies a random relative SNR offset between the two sources.
  6. Sums to create the mixture (mixture = s1 + s2 by construction).
  7. Peak-normalises using the mixture peak.

This gives effectively infinite training combinations from the same source
utterances, compared to 13,900 fixed pairs in pre-generated Libri2Mix.

Validation still uses the pre-generated LibriMixDataset for fair comparison.
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SR       = 16_000
DEFAULT_DURATION = 4.0
DEFAULT_CLIP_LEN = int(DEFAULT_SR * DEFAULT_DURATION)   # 64,000 samples


def _load_wav(path: str) -> Tensor:
    wav, sr = torchaudio.load(path)
    if sr != DEFAULT_SR:
        raise ValueError(f"Expected {DEFAULT_SR} Hz, got {sr} Hz: {path}")
    return wav.mean(0).float()


def _speed_perturb(wav: Tensor) -> Tensor:
    """Apply random speed perturbation (0.95x–1.05x) via linear interpolation."""
    speed = random.uniform(0.95, 1.05)
    if abs(speed - 1.0) < 0.005:
        return wav
    new_len = int(wav.shape[0] / speed)
    return F.interpolate(
        wav.unsqueeze(0).unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0).squeeze(0)


def _crop_or_pad(wav: Tensor, target_len: int) -> Tensor:
    n = wav.shape[0]
    if n >= target_len:
        start = random.randint(0, n - target_len)
        return wav[start : start + target_len]
    reps = math.ceil(target_len / n)
    return wav.repeat(reps)[:target_len]


def spike_safe_augment(
    mixture: Tensor, s1: Tensor, s2: Tensor, L: int
) -> Tuple[Tensor, Tensor, Tensor]:
    if random.random() < 0.5:
        mixture, s1, s2 = -mixture, -s1, -s2

    if random.random() < 0.5:
        shift = random.randint(1, L // 4)
        mixture = torch.roll(mixture, shift)
        s1      = torch.roll(s1,      shift)
        s2      = torch.roll(s2,      shift)

    if random.random() < 0.6:
        level = random.uniform(0.0, 0.015)
        rms   = mixture.pow(2).mean().sqrt().clamp(min=1e-8)
        mixture = mixture + torch.randn_like(mixture) * level * rms

    if random.random() < 0.3:
        taper_len = random.randint(L // 16, L // 8)
        start     = random.randint(0, L - taper_len)
        t         = torch.linspace(0.0, 2.0 * math.pi, taper_len)
        envelope  = torch.ones(L)
        envelope[start : start + taper_len] = 0.5 - 0.5 * torch.cos(t)
        mixture, s1, s2 = mixture * envelope, s1 * envelope, s2 * envelope

    return mixture, s1, s2


class DynamicMixDataset(Dataset):
    """
    On-the-fly mixing of clean speech utterances.

    Pools all .wav files from s1/ and s2/ in the given split directory.
    Each __getitem__ randomly pairs two utterances, optionally speed-perturbs,
    crops/pads, applies relative SNR jitter, and sums to create the mixture.
    """

    def __init__(
        self,
        root:          str,
        split:         str   = "train-100",
        clip_len:      int   = DEFAULT_CLIP_LEN,
        epoch_size:    int   = 13_900,
        speed_perturb: bool  = True,
        snr_range_db:  Tuple[float, float] = (-5.0, 5.0),
        augment:       bool  = True,
    ) -> None:
        split_dir = Path(root) / split
        s1_dir    = split_dir / "s1"
        s2_dir    = split_dir / "s2"

        for d in (s1_dir, s2_dir):
            if not d.is_dir():
                raise RuntimeError(f"Directory not found: {d}")

        self.utterances: List[str] = []
        for d in (s1_dir, s2_dir):
            self.utterances.extend(str(p) for p in sorted(d.glob("*.wav")))

        if not self.utterances:
            raise RuntimeError(f"No .wav files found in {s1_dir} or {s2_dir}")

        self.clip_len      = clip_len
        self.epoch_size     = epoch_size
        self.speed_perturb  = speed_perturb
        self.snr_range_db   = snr_range_db
        self.augment        = augment

        print(f"[DynamicMix] Pooled {len(self.utterances)} utterances from {split}")
        print(f"[DynamicMix] epoch_size={epoch_size}  speed_perturb={speed_perturb}  "
              f"snr_range={snr_range_db}")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        i1, i2 = random.sample(range(len(self.utterances)), 2)

        s1 = _load_wav(self.utterances[i1])
        s2 = _load_wav(self.utterances[i2])

        if self.speed_perturb:
            s1 = _speed_perturb(s1)
            s2 = _speed_perturb(s2)

        s1 = _crop_or_pad(s1, self.clip_len)
        s2 = _crop_or_pad(s2, self.clip_len)

        snr_db = random.uniform(*self.snr_range_db)
        gain   = 10.0 ** (snr_db / 20.0)
        s2     = s2 * gain

        mixture = s1 + s2

        peak = mixture.abs().max().clamp(min=1e-8)
        mixture, s1, s2 = mixture / peak, s1 / peak, s2 / peak

        if self.augment:
            mixture, s1, s2 = spike_safe_augment(mixture, s1, s2, self.clip_len)

        return mixture, s1, s2


def build_dynamic_mix_dataloaders(
    librimix_root: str,
    train_split:   str = "train-100",
    val_split:     str = "dev",
    clip_len:      int = DEFAULT_CLIP_LEN,
    batch_size:    int = 8,
    num_workers:   int = 8,
    epoch_size:    int = 13_900,
    speed_perturb: bool = True,
    snr_range_db:  Tuple[float, float] = (-5.0, 5.0),
    train_augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    from librimix_dataset import LibriMixDataset

    train_ds = DynamicMixDataset(
        root          = librimix_root,
        split         = train_split,
        clip_len      = clip_len,
        epoch_size    = epoch_size,
        speed_perturb = speed_perturb,
        snr_range_db  = snr_range_db,
        augment       = train_augment,
    )
    val_ds = LibriMixDataset(librimix_root, val_split, clip_len, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        persistent_workers = num_workers > 0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 dynamic_mix_dataset.py <librimix_root>")
        sys.exit(1)

    root = sys.argv[1]
    print(f"Building dynamic mix dataloaders from: {root}")

    train_dl, val_dl = build_dynamic_mix_dataloaders(
        librimix_root = root,
        batch_size    = 4,
        num_workers   = 0,
        epoch_size    = 100,
    )

    print(f"Train: {len(train_dl.dataset)} items/epoch  ({len(train_dl)} batches)")
    print(f"Val:   {len(val_dl.dataset)} files  ({len(val_dl)} batches)")

    mix, s1, s2 = next(iter(train_dl))
    print(f"\nBatch shapes — mix: {tuple(mix.shape)}  s1: {tuple(s1.shape)}")
    print(f"Mix range    — min: {mix.min():.4f}  max: {mix.max():.4f}")

    recon_err = (mix - s1 - s2).abs().mean().item()
    print(f"mean|mix - s1 - s2|: {recon_err:.6f}  (should be ~0 before augment)")

    print("OK")
