"""
dataset.py – PyTorch Dataset for the binaural sound-source-separation corpus.

Directory layout expected under `data_root`:
    data_root/
        <filename>.wav          # all audio clips (sources, transients, dynamic)

The Dataset:
  1. Reads labels.csv + original_data_map.json to build per-scene metadata.
  2. At __getitem__, constructs a mixed binaural waveform on-the-fly by:
       sum(source_clips) + sum(transient_clips) → convolved with the BRIR clip
     (or loads a pre-mixed file if `use_premixed=True`).
  3. Returns (waveform [C, T], multi-hot label [8]).
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    AudioConfig, ALL_LABELS, SOURCE_LABELS, TRANSIENT_LABELS,
    audio_cfg, train_cfg
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_wav(path: str, target_sr: int = 16_000, target_channels: int = 2) -> torch.Tensor:
    """Load a WAV file, resample, and ensure `target_channels` channels.
    Returns tensor of shape (C, T).
    """
    waveform, sr = torchaudio.load(path)                      # (C, T_orig)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    # Channel normalisation
    if waveform.shape[0] == 1 and target_channels == 2:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > target_channels:
        waveform = waveform[:target_channels]
    return waveform


def pad_or_trim(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Force waveform to exactly n_samples along dim -1."""
    T = waveform.shape[-1]
    if T >= n_samples:
        return waveform[..., :n_samples]
    return torch.nn.functional.pad(waveform, (0, n_samples - T))


def mix_waveforms(waves: List[torch.Tensor], rms_normalise: bool = True) -> torch.Tensor:
    """Sum a list of (C, T) waveforms with optional RMS normalisation per source."""
    if not waves:
        raise ValueError("mix_waveforms: empty list")
    if rms_normalise:
        normalised = []
        for w in waves:
            rms = w.pow(2).mean().sqrt().clamp(min=1e-8)
            normalised.append(w / rms)
        waves = normalised
    mix = torch.stack(waves, dim=0).sum(dim=0)               # (C, T)
    # Peak normalise the mix
    peak = mix.abs().max().clamp(min=1e-8)
    return mix / peak


def labels_to_multihot(source_labels: List[str],
                        transient_labels: List[str]) -> torch.Tensor:
    """Return an 8-dimensional multi-hot tensor (source + transient classes)."""
    vec = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
    for lbl in source_labels:
        if lbl in SOURCE_LABELS:
            vec[SOURCE_LABELS.index(lbl)] = 1.0
    for lbl in transient_labels:
        if lbl in TRANSIENT_LABELS:
            vec[len(SOURCE_LABELS) + TRANSIENT_LABELS.index(lbl)] = 1.0
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Scene metadata
# ──────────────────────────────────────────────────────────────────────────────

class SceneMeta:
    """Lightweight container for one scene's file paths and labels."""

    def __init__(self, scene_dict: dict, label_row: pd.Series, data_root: str):
        self.id              = scene_dict["id"]
        self.data_root       = Path(data_root)
        self.source_clips    = scene_dict.get("source_clips", [])
        self.transient_clips = scene_dict.get("transient_clips", [])
        self.dynamic_clips   = scene_dict.get("dynamic_clips", [])

        # Parsed label lists from CSV
        src_str  = label_row["source_labels"]
        trn_str  = label_row["transient_labels"]
        self.source_label_list   = [s.strip() for s in src_str.split(",")]  if pd.notna(src_str) else []
        self.transient_label_list= [s.strip() for s in trn_str.split(",")]  if pd.notna(trn_str) else []

    CLIP_SUBDIRS = {
        "source":    "sources",
        "transient": "transients",
        "dynamic":   "dynamic",
    }

    def resolve(self, clip_dict: dict, clip_type: str = "source") -> Optional[Path]:
        fname  = Path(clip_dict["path"]).name
        subdir = self.CLIP_SUBDIRS.get(clip_type, "sources")
        # Check expected subdir first, then shared/ as fallback
        for folder in (subdir, "shared"):
            candidate = self.data_root / folder / fname
            if candidate.exists():
                return candidate
        return None

    @property
    def multihot(self) -> torch.Tensor:
        return labels_to_multihot(self.source_label_list, self.transient_label_list)

    @property
    def n_sources(self) -> int:
        return len(self.source_clips)

    @property
    def has_transients(self) -> bool:
        return len(self.transient_clips) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SoundSeparationDataset(Dataset):
    """
    Binaural sound source separation dataset.

    Parameters
    ----------
    labels_csv : str
        Path to labels.csv
    data_map_json : str
        Path to original_data_map.json
    data_root : str
        Root directory where .wav files live
    audio_cfg : AudioConfig
        Audio configuration
    use_premixed : bool
        If True and a pre-mixed file exists in `data_root/mixed/`, load it
        directly instead of building the mix on-the-fly.
    augment : bool
        Apply random gain, time-shift, and additive noise augmentations.
    transform : callable, optional
        Additional transform applied to the final waveform tensor.
    """

    def __init__(
        self,
        labels_csv:   str,
        data_map_json: str,
        data_root:    str,
        audio_cfg:    AudioConfig  = audio_cfg,
        use_premixed: bool         = False,
        augment:      bool         = False,
        transform                  = None,
    ):
        self.audio_cfg    = audio_cfg
        self.use_premixed = use_premixed
        self.augment      = augment
        self.transform    = transform
        self.data_root    = Path(data_root)

        # ── Load metadata ──────────────────────────────────────────────────
        labels_df = pd.read_csv(labels_csv, index_col="id")
        with open(data_map_json) as f:
            data_map = json.load(f)

        self.scenes: List[SceneMeta] = []
        for scene_dict in data_map["data"]:
            sid = scene_dict["id"]
            if sid in labels_df.index:
                row = labels_df.loc[sid]
                self.scenes.append(SceneMeta(scene_dict, row, data_root))

        print(f"[Dataset] Loaded {len(self.scenes)} scenes from {data_root}")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict:
        scene = self.scenes[idx]
        waveform = self._build_waveform(scene)

        if self.augment:
            waveform = self._augment(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,               # (C=2, T=48000)
            "label":    scene.multihot,         # (8,) float multi-hot
            "scene_id": scene.id,
            "n_sources": scene.n_sources,
        }

    # ── Waveform construction ─────────────────────────────────────────────────

    def _build_waveform(self, scene: SceneMeta) -> torch.Tensor:
        """Build a mixed stereo waveform for this scene."""
        sr   = self.audio_cfg.sample_rate
        ns   = self.audio_cfg.n_samples
        nchan = self.audio_cfg.channels

        # Try pre-mixed first
        if self.use_premixed:
            premix_path = self.data_root / "mixed" / f"{scene.id:05d}.wav"
            if premix_path.exists():
                return pad_or_trim(load_wav(str(premix_path), sr, nchan), ns)

        # Fall back: sum source + transient clips
        waves = []
        typed_clips = (
            [(c, "source")    for c in scene.source_clips] +
            [(c, "transient") for c in scene.transient_clips]
        )
        for clip, clip_type in typed_clips:
            p = scene.resolve(clip, clip_type)
            if p is not None:
                try:
                    w = load_wav(str(p), sr, nchan)
                    waves.append(pad_or_trim(w, ns))
                except Exception as e:
                    print(f"[Dataset] Warning: could not load {p}: {e}")

        if not waves:
            # Return silent tensor if no files found (dataset not yet downloaded)
            return torch.zeros(nchan, ns)

        return mix_waveforms(waves)

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random gain ±6 dB
        gain = 10 ** (random.uniform(-6, 6) / 20)
        waveform = waveform * gain

        # Random time shift (up to 500 ms)
        max_shift = int(self.audio_cfg.sample_rate * 0.5)
        shift = random.randint(-max_shift, max_shift)
        waveform = torch.roll(waveform, shifts=shift, dims=-1)

        # Additive white Gaussian noise (SNR 30–60 dB)
        snr_db = random.uniform(30, 60)
        signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_rms  = signal_rms / (10 ** (snr_db / 20))
        waveform   = waveform + noise_rms * torch.randn_like(waveform)

        # Re-peak normalise
        peak = waveform.abs().max().clamp(min=1e-8)
        return waveform / peak


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    labels_csv:    str,
    data_map_json: str,
    data_root:     str,
    audio_cfg:     AudioConfig = audio_cfg,
    batch_size:    int         = 16,
    num_workers:   int         = 4,
    val_split:     float       = 0.15,
    test_split:    float       = 0.10,
    seed:          int         = 42,
    use_premixed:  bool        = False,
    transform                  = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Augmentation is only applied to the training split.
    """
    generator = torch.Generator().manual_seed(seed)

    full_dataset = SoundSeparationDataset(
        labels_csv, data_map_json, data_root,
        audio_cfg=audio_cfg, use_premixed=use_premixed,
        augment=False, transform=transform,
    )
    n = len(full_dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    # Wrap train split with augmentation
    train_ds.dataset = SoundSeparationDataset(
        labels_csv, data_map_json, data_root,
        audio_cfg=audio_cfg, use_premixed=use_premixed,
        augment=True, transform=transform,
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds,   shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )