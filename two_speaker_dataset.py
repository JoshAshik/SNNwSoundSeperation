"""
two_speaker_dataset.py — On-the-fly 2-speaker mixture generator for PIT pre-training.

Returns (mixture, target_1, target_2) — all shape (L,), peak-normalised.
Designed as a pre-training / curriculum stage before the 8-channel FSD50K fine-tune.

Usage:
    from two_speaker_dataset import TwoSpeakerMixDataset, build_two_speaker_dataloaders

    train_dl, val_dl = build_two_speaker_dataloaders(fsd50k_root="./data/fsd50k")

See docs/dataset.md §Two-Speaker Mixture Dataset for design rationale.
"""

import math
import os
import random
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ── resample kernel cache (44.1→16kHz computed once) ─────────────────────────
_RESAMPLER_CACHE: dict = {}


# ── low-level helpers ─────────────────────────────────────────────────────────

def _get_resampler(orig_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    key = (orig_sr, target_sr)
    if key not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
    return _RESAMPLER_CACHE[key]


def load_and_fit(path: str, target_len: int, sr: int = 16000) -> Tensor:
    """
    Load a WAV file, resample to sr, convert to mono, then either
    random-crop (if longer) or tile-pad (if shorter) to exactly target_len samples.
    Returns a 1-D float32 tensor of shape (target_len,).
    """
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(0)  # stereo → mono
    if orig_sr != sr:
        resampler = _get_resampler(orig_sr, sr)
        wav = resampler(wav)
    n = wav.shape[0]
    if n >= target_len:
        start = random.randint(0, n - target_len)
        wav = wav[start : start + target_len]
    else:
        reps = math.ceil(target_len / n)
        wav = wav.repeat(reps)[:target_len]
    return wav.float()


def scale_to_snr(signal: Tensor, noise: Tensor, snr_db: float) -> Tensor:
    """
    Return `noise` scaled so that SNR(signal, result) == snr_db.

    snr_db > 0  →  result is quieter than signal
    snr_db < 0  →  result is louder than signal
    snr_db = 0  →  result has the same RMS as signal
    """
    eps = 1e-8
    sig_rms = signal.pow(2).mean().clamp(min=eps).sqrt()
    noi_rms = noise.pow(2).mean().clamp(min=eps).sqrt()
    target_rms = sig_rms * (10 ** (-snr_db / 20.0))
    return noise * (target_rms / noi_rms)


def place_with_overlap(s1: Tensor, s2: Tensor, overlap_ratio: float) -> Tensor:
    """
    Place s2 into a zero tensor of the same length as s1 such that
    `overlap_ratio` fraction of s2 overlaps with s1.

    overlap_ratio=1.0  →  s2 starts at sample 0  (fully simultaneous)
    overlap_ratio=0.0  →  s2 starts at or after s1 ends (fully sequential,
                          output is truncated to len(s1) so trailing part is lost)

    A random jitter within the allowed offset range is applied on every call
    so that repeated scenes at the same overlap_ratio still differ.

    Returns a 1-D tensor of shape (len(s1),).
    """
    L = len(s1)
    # max_start: furthest right s2 can start while keeping overlap >= overlap_ratio
    max_start = int(L * (1.0 - overlap_ratio))
    start = random.randint(0, max(max_start, 0))
    placed = torch.zeros(L, dtype=s2.dtype)
    copy_len = min(L - start, L)
    placed[start : start + copy_len] = s2[:copy_len]
    return placed


# ── spike-safe augmentation ───────────────────────────────────────────────────

def spike_safe_augment(
    mixture: Tensor,
    s1: Tensor,
    s2: Tensor,
    L: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Apply augmentations that are safe for SNN (LIF neuron) inputs.

    Rule: any operation applied to `mixture` is also applied identically to
    s1 and s2 so that PIT targets stay aligned.

    Safe operations preserve temporal continuity in the waveform — abrupt
    discontinuities create high-energy transients in the Conv1d encoder output
    that cause spurious LIF spike bursts unrelated to any audio event.

    NOT applied here (reasons in docs/dataset.md):
      - Hard rectangular zero-masks
      - Waveform-domain SpecAugment
      - Phase scrambling / aggressive pitch shift
    """

    # ── 1. Gain jitter ±6 dB (uniform scale, no discontinuity) ───────────────
    if random.random() < 0.8:
        gain_db = random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20.0)
        mixture = mixture * gain
        s1      = s1      * gain
        s2      = s2      * gain

    # ── 2. Polarity inversion (uniform flip; Conv1d encoder is polarity-aware) ─
    if random.random() < 0.5:
        mixture = -mixture
        s1      = -s1
        s2      = -s2

    # ── 3. Circular time shift ≤0.75 s (torch.roll wraps cleanly, no edge) ───
    if random.random() < 0.5:
        max_shift = L // 4   # 0.75 s at 16 kHz for L=48000
        shift = random.randint(1, max_shift)
        mixture = torch.roll(mixture, shift)
        s1      = torch.roll(s1,      shift)
        s2      = torch.roll(s2,      shift)

    # ── 4. Low-level additive noise (mixture only; targets stay clean for PIT) ─
    if random.random() < 0.6:
        level = random.uniform(0.0, 0.015)
        rms   = mixture.pow(2).mean().sqrt().clamp(min=1e-8)
        mixture = mixture + torch.randn_like(mixture) * level * rms

    # ── 5. Smooth cosine amplitude taper (soft dip, no step edge) ─────────────
    # Models a brief fade — encoder sees a gradual amplitude change, not a cliff.
    if random.random() < 0.3:
        taper_len = random.randint(L // 16, L // 8)   # 188–375 samples @ 16kHz
        start     = random.randint(0, L - taper_len)
        # Cosine ramp: 1 → 0 → 1 over taper_len samples
        t        = torch.linspace(0.0, 2.0 * math.pi, taper_len)
        taper    = 0.5 - 0.5 * torch.cos(t)           # shape (taper_len,)
        envelope = torch.ones(L)
        envelope[start : start + taper_len] = taper
        mixture = mixture * envelope
        s1      = s1      * envelope
        s2      = s2      * envelope

    return mixture, s1, s2


# ── Dataset ───────────────────────────────────────────────────────────────────

class TwoSpeakerMixDataset(Dataset):
    """
    On-the-fly 2-speaker mixture generator.

    Each call to __getitem__ picks two clips at random, scales the second to
    a target SNR, places it with a random overlap, and returns the peak-
    normalised (mixture, target_1, target_2) triple.

    Compatible with any DataLoader num_workers value (no shared state).
    """

    def __init__(
        self,
        clip_index: List[Tuple[str, str]],
        scenes_per_epoch: int = 5000,
        sr: int = 16000,
        duration: float = 3.0,
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        snr_range: Tuple[float, float] = (-5.0, 5.0),
        augment: bool = True,
    ) -> None:
        """
        Args:
            clip_index:       List of (wav_path, label_str) pairs.
            scenes_per_epoch: Number of synthetic scenes per epoch (Dataset length).
            sr:               Target sample rate in Hz.
            duration:         Clip duration in seconds.
            overlap_range:    (min_overlap, max_overlap) as fractions [0, 1].
                              1.0 = fully simultaneous, 0.0 = fully sequential.
            snr_range:        (min_snr_db, max_snr_db). Positive = s1 louder.
            augment:          If True, apply spike-safe augmentations.
        """
        if len(clip_index) < 2:
            raise ValueError("clip_index must contain at least 2 clips.")
        self.clips = clip_index
        self.N = scenes_per_epoch
        self.sr = sr
        self.L = int(sr * duration)
        self.overlap_range = overlap_range
        self.snr_range = snr_range
        self.augment = augment

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        # ── pick two distinct clips ──────────────────────────────────────────
        i1 = idx % len(self.clips)
        i2 = random.randint(0, len(self.clips) - 1)
        while i2 == i1:
            i2 = random.randint(0, len(self.clips) - 1)

        s1 = load_and_fit(self.clips[i1][0], self.L, self.sr)
        s2 = load_and_fit(self.clips[i2][0], self.L, self.sr)

        # v6: RMS-normalise each source to unit energy before SNR scaling.
        # Without this, scale_to_snr balances relative to s1's raw RMS —
        # clips with different recording levels produce inconsistent mixture
        # energies, and the model learns to exploit loudness as a shortcut
        # (assigning the louder source to one fixed channel, collapsing the
        # other speaker's output to near-zero on quieter inputs).
        # Normalising to RMS=1.0 first ensures the ±5 dB SNR range maps to
        # a consistent absolute energy and removes the loudness shortcut.
        s1 = s1 / s1.pow(2).mean().clamp(min=1e-8).sqrt()
        s2 = s2 / s2.pow(2).mean().clamp(min=1e-8).sqrt()

        # ── SNR balancing ────────────────────────────────────────────────────
        snr_db = random.uniform(*self.snr_range)
        s2 = scale_to_snr(s1, s2, snr_db)

        # ── temporal placement ───────────────────────────────────────────────
        overlap_ratio = random.uniform(*self.overlap_range)
        s2_placed = place_with_overlap(s1, s2, overlap_ratio)

        # ── mix and peak-normalise (targets share the same scale factor) ─────
        mixture = s1 + s2_placed
        peak = mixture.abs().max().clamp(min=1e-8)
        mixture  = mixture   / peak
        s1_norm  = s1        / peak
        s2_norm  = s2_placed / peak

        # ── spike-safe augmentation ──────────────────────────────────────────
        if self.augment:
            mixture, s1_norm, s2_norm = spike_safe_augment(
                mixture, s1_norm, s2_norm, self.L
            )

        return mixture, s1_norm, s2_norm


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_two_speaker_dataloaders(
    fsd50k_root: str,
    scenes_per_epoch: int = 5000,
    val_scenes: int = 1000,
    batch_size: int = 8,
    num_workers: int = 4,
    sr: int = 16000,
    overlap_range: Tuple[float, float] = (0.3, 1.0),
    snr_range: Tuple[float, float] = (-5.0, 5.0),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from an FSD50K root directory.

    Reads FSD50K.ground_truth/dev.csv to get per-clip paths and splits.
    Val uses a fixed seed so scenes are reproducible across runs.

    Args:
        fsd50k_root:      Path to FSD50K root (contains FSD50K.dev_audio/ etc.)
        scenes_per_epoch: Training dataset length.
        val_scenes:       Validation dataset length.
        batch_size:       Batch size for both loaders.
        num_workers:      DataLoader worker count.
        sr:               Target sample rate.
        overlap_range:    Passed to TwoSpeakerMixDataset.
        snr_range:        Passed to TwoSpeakerMixDataset.
        seed:             RNG seed for val set (train is always random).

    Returns:
        (train_loader, val_loader)
    """
    import pandas as pd

    dev_csv  = os.path.join(fsd50k_root, "FSD50K.ground_truth", "dev.csv")
    audio_dir = os.path.join(fsd50k_root, "FSD50K.dev_audio")

    df = pd.read_csv(dev_csv)

    def _build_index(split_df) -> List[Tuple[str, str]]:
        index = []
        for _, row in split_df.iterrows():
            path = os.path.join(audio_dir, f"{row['fname']}.wav")
            if os.path.exists(path):
                labels = str(row.get("labels", "unknown"))
                index.append((path, labels))
        return index

    train_idx = _build_index(df[df["split"] == "train"])
    val_idx   = _build_index(df[df["split"] == "val"])

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"No clips found under {audio_dir}. "
            "Check fsd50k_root and that FSD50K.dev_audio/ is present."
        )

    train_ds = TwoSpeakerMixDataset(
        clip_index=train_idx,
        scenes_per_epoch=scenes_per_epoch,
        sr=sr,
        overlap_range=overlap_range,
        snr_range=snr_range,
        augment=True,
    )

    # Val: fix the RNG seed so scenes are identical across runs
    rng_state = random.getstate()
    random.seed(seed)
    val_ds = TwoSpeakerMixDataset(
        clip_index=val_idx,
        scenes_per_epoch=val_scenes,
        sr=sr,
        overlap_range=overlap_range,
        snr_range=snr_range,
        augment=False,
    )
    random.setstate(rng_state)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python3 two_speaker_dataset.py <fsd50k_root>")
        sys.exit(1)

    fsd50k_root = sys.argv[1]
    print(f"Building dataloaders from: {fsd50k_root}")

    train_dl, val_dl = build_two_speaker_dataloaders(
        fsd50k_root=fsd50k_root,
        scenes_per_epoch=100,
        val_scenes=20,
        batch_size=4,
        num_workers=0,
    )

    print(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    mix, t1, t2 = next(iter(train_dl))
    print(f"Batch shapes  — mixture: {tuple(mix.shape)}, t1: {tuple(t1.shape)}, t2: {tuple(t2.shape)}")
    print(f"Mixture range — min: {mix.min():.4f}  max: {mix.max():.4f}")

    # Verify peak normalisation: mixture peak should be <= 1.0
    peaks = mix.abs().flatten(1).max(dim=1).values
    print(f"Per-sample peaks (should be ~1.0): {peaks.tolist()}")

    # Verify targets sum to mixture (within floating point)
    recon_err = (mix - t1 - t2).abs().mean().item()
    print(f"Mean |mixture - t1 - t2|: {recon_err:.6f}  (should be ~0)")

    print("OK")
