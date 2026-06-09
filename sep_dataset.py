"""
sep_dataset.py — Dataset with repeat-factor expansion and SpecAugment.

Key additions vs previous version:
  1. repeat_factor (default 4):
       When augment=True, each scene appears repeat_factor times per epoch,
       each time with a DIFFERENT random augmentation. With 300 training
       scenes and repeat_factor=4, the model takes 1,200 gradient steps per
       epoch instead of 300 — without seeing any new real data. This is the
       single most impactful change for a small dataset.

  2. SpecAugment (applied to mixture only):
       Randomly zeros out frequency bands and time windows in the STFT domain
       of the input mixture. The target sources remain clean. This forces the
       model to learn robust separation rather than memorising spectral
       fingerprints of specific training recordings. Fast CPU operation —
       no phase vocoder needed.
       - freq_mask:  up to 20 frequency bins (~300 Hz) zeroed
       - time_mask:  up to 800 frames (~0.3 s) zeroed

Summary of all augmentations (training only):
  1. Synthetic remixing (50%): swap transient clips from another scene
  2. Gain (100%): ±6 dB random scaling
  3. Noise injection (100%): 0–2% noise to mixture only
  4. Time shift (50%): circular roll up to 0.5 s
  5. SpecAugment (60%): freq + time masking on mixture STFT
  6. Repeat factor (4×): each scene appears 4 times per epoch with
       independent random augmentation choices
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from config import AudioConfig, ALL_LABELS, SOURCE_LABELS, TRANSIENT_LABELS, audio_cfg

# Fixed label-to-channel mapping for 8-source separation.
# Channel index corresponds to label index in ALL_LABELS.
SEP_LABELS    = ALL_LABELS   # ["human_voice","animal","impacts","music","weather","alerts","domestic_and_objects","tools_and_machines"]
N_SEP_SOURCES = len(SEP_LABELS)  # 8


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str, target_sr: int = 16_000, channels: int = 1) -> torch.Tensor:
    """Load wav → (channels, T), resampled to target_sr."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = T.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    if wav.shape[0] == 1 and channels == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > channels:
        wav = wav[:channels]
    return wav  # (C, T)


def pad_or_trim(wav: torch.Tensor, n: int) -> torch.Tensor:
    L = wav.shape[-1]
    if L >= n:
        return wav[..., :n]
    return F.pad(wav, (0, n - L))


def rms_normalise(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp(min=eps)
    return wav / rms


# ─────────────────────────────────────────────────────────────────────────────
# Separation Dataset
# ─────────────────────────────────────────────────────────────────────────────

CLIP_SUBDIRS = {
    "source":    "sources",
    "transient": "transients",
    "dynamic":   "dynamic",
}


def spec_augment(
    mixture: torch.Tensor,
    sr: int,
    freq_mask_param: int = 20,
    time_mask_param: int = 800,
    n_fft: int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    """
    Apply SpecAugment to a mono mixture waveform.

    Operates in the STFT domain: zeros random frequency bands and time
    windows then reconstructs with ISTFT. Applied to the mixture only —
    target sources stay clean, so the model must learn robust separation
    rather than memorising spectral fingerprints.

    Args:
        mixture:         (T,) or (1, T) waveform, values in [-1, 1]
        sr:              sample rate (unused, kept for API clarity)
        freq_mask_param: max frequency bins to zero per mask
        time_mask_param: max time frames to zero per mask
        n_fft:           STFT window size
        hop_length:      STFT hop

    Returns:
        Augmented waveform, same shape as input, clamped to [-1, 1].
    """
    squeeze = mixture.dim() == 1
    if squeeze:
        mixture = mixture.unsqueeze(0)          # (1, T)

    T_orig = mixture.shape[-1]
    window = torch.hann_window(n_fft, device=mixture.device)

    # Compute STFT for each channel (here always 1)
    spec = torch.stft(
        mixture[0], n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True,
    )                                            # (F, T_frames)

    F, T_frames = spec.shape

    # ── Frequency masking ───────────────────────────────────────────────────
    f_width = random.randint(0, freq_mask_param)
    f_start = random.randint(0, max(0, F - f_width))
    spec[f_start:f_start + f_width, :] = 0.0

    # ── Time masking ────────────────────────────────────────────────────────
    t_width = random.randint(0, min(time_mask_param, T_frames))
    t_start = random.randint(0, max(0, T_frames - t_width))
    spec[:, t_start:t_start + t_width] = 0.0

    # ── Reconstruct waveform ────────────────────────────────────────────────
    wav_out = torch.istft(
        spec, n_fft=n_fft, hop_length=hop_length,
        window=window, length=T_orig,
    )                                            # (T,)

    wav_out = wav_out.unsqueeze(0).clamp(-1.0, 1.0)   # (1, T)
    return wav_out.squeeze(0) if squeeze else wav_out


class SoundSepDataset(Dataset):
    """
    Separation dataset built from the 400-scene binaural corpus.

    Returns per scene:
        mixture:  (T,)   — mono mix, peak-normalised
        sources:  (8, T) — one channel per SEP_LABELS category (ALL_LABELS order);
                           channels for absent categories are zero
        labels:   (8,)   — multi-hot label vector
        scene_id: int

    When augment=True and repeat_factor > 1, the logical dataset length is
    len(scenes) * repeat_factor. Each "copy" of a scene uses independent
    random augmentation, so the model sees the same recording with a
    different gain, time-shift, remix, and spectral mask every time.
    """

    CHANNELS       = 1      # mono
    REMIX_PROB     = 0.5    # cross-scene transient remixing probability
    SPEC_AUG_PROB  = 0.6    # probability of applying SpecAugment to mixture

    def __init__(
        self,
        labels_csv:    str,
        data_map_json: str,
        data_root:     str,
        audio_cfg:     AudioConfig = audio_cfg,
        augment:       bool        = False,
        repeat_factor: int         = 1,
    ):
        self.data_root     = Path(data_root)
        self.audio_cfg     = audio_cfg
        self.n_samples     = audio_cfg.n_samples
        self.sr            = audio_cfg.sample_rate
        self.augment       = augment
        self.repeat_factor = repeat_factor if augment else 1

        labels_df = pd.read_csv(labels_csv, index_col="id")
        with open(data_map_json) as f:
            raw = json.load(f)

        self.scenes = []
        for scene in raw["data"]:
            sid = scene["id"]
            if sid not in labels_df.index:
                continue
            row = labels_df.loc[sid]

            label = torch.zeros(len(ALL_LABELS))
            src_str = row["source_labels"]
            trn_str = row["transient_labels"]
            if pd.notna(src_str):
                for lbl in src_str.split(","):
                    lbl = lbl.strip()
                    if lbl in SOURCE_LABELS:
                        label[SOURCE_LABELS.index(lbl)] = 1.0
            if pd.notna(trn_str):
                for lbl in trn_str.split(","):
                    lbl = lbl.strip()
                    if lbl in TRANSIENT_LABELS:
                        label[len(SOURCE_LABELS) + TRANSIENT_LABELS.index(lbl)] = 1.0

            self.scenes.append({
                "id":              sid,
                "source_clips":    scene.get("source_clips", []),
                "transient_clips": scene.get("transient_clips", []),
                "label":           label,
            })

        eff_len = len(self.scenes) * self.repeat_factor
        print(f"[SepDataset] {len(self.scenes)} scenes × {self.repeat_factor} repeats "
              f"= {eff_len} steps/epoch | "
              f"sr={self.sr} Hz | duration={self.n_samples/self.sr:.1f}s | "
              f"augment={augment}")

    def __len__(self):
        return len(self.scenes) * self.repeat_factor

    def _load_clips(self, clips: List[dict], clip_type: str) -> List[torch.Tensor]:
        loaded = []
        subdir = CLIP_SUBDIRS.get(clip_type, "sources")
        for clip in clips:
            fname = Path(clip["path"]).name
            for folder in (subdir, "shared"):
                p = self.data_root / folder / fname
                if p.exists():
                    try:
                        w = load_wav(str(p), self.sr, self.CHANNELS)
                        w = pad_or_trim(w, self.n_samples)
                        loaded.append(w)
                    except Exception:
                        pass
                    break
        return loaded

    def _sum_clips(self, clips: List[torch.Tensor]) -> torch.Tensor:
        if not clips:
            return torch.zeros(self.CHANNELS, self.n_samples)
        normed = [rms_normalise(c) for c in clips]
        return torch.stack(normed).sum(dim=0)

    def _pick_other_idx(self, idx: int) -> int:
        other = idx
        while other == idx:
            other = random.randint(0, len(self.scenes) - 1)
        return other

    def _clips_by_label(self, all_clips: list, clip_type: str) -> dict:
        """Group a flat list of clip dicts by unified_label, loading each wav."""
        by_label: dict = {lbl: [] for lbl in SEP_LABELS}
        for clip in all_clips:
            lbl = clip.get("unified_label", "")
            if lbl not in by_label:
                continue
            fname = Path(clip["path"]).name
            subdir = CLIP_SUBDIRS.get(clip_type, "sources")
            for folder in (subdir, "shared"):
                p = self.data_root / folder / fname
                if p.exists():
                    try:
                        w = load_wav(str(p), self.sr, self.CHANNELS)
                        w = pad_or_trim(w, self.n_samples)
                        by_label[lbl].append(w)
                    except Exception:
                        pass
                    break
        return by_label

    def __getitem__(self, idx: int) -> Dict:
        # Modulo maps repeated indices back to real scenes.
        # Each pass through the same scene gets independent random augmentation.
        scene = self.scenes[idx % len(self.scenes)]

        # ── 1. Synthetic per-category remixing ────────────────────────────────
        # Swap clips for a single randomly chosen label category from a donor scene.
        # This creates fine-grained augmentation (n_train^2 combinations per category).
        swap_label = None
        donor_clips: dict = {}
        if self.augment and random.random() < self.REMIX_PROB:
            donor      = self.scenes[self._pick_other_idx(idx % len(self.scenes))]
            swap_label = random.choice(SEP_LABELS)
            # Collect donor's clips for the swapped label (from either source or transient list)
            donor_all  = donor["source_clips"] + donor["transient_clips"]
            for clip in donor_all:
                if clip.get("unified_label") == swap_label:
                    donor_clips.setdefault(swap_label, []).append(clip)

        # Load all scene clips grouped by label
        all_scene_clips = scene["source_clips"] + scene["transient_clips"]
        # Determine clip_type hint per clip (used for subdirectory lookup)
        src_set = {Path(c["path"]).name for c in scene["source_clips"]}
        by_label: dict = {lbl: [] for lbl in SEP_LABELS}
        for clip in all_scene_clips:
            lbl = clip.get("unified_label", "")
            if lbl not in by_label:
                continue
            # Skip this label if we're swapping it from donor
            if lbl == swap_label:
                continue
            ctype = "source" if Path(clip["path"]).name in src_set else "transient"
            fname = Path(clip["path"]).name
            subdir = CLIP_SUBDIRS.get(ctype, "sources")
            for folder in (subdir, "shared"):
                p = self.data_root / folder / fname
                if p.exists():
                    try:
                        w = load_wav(str(p), self.sr, self.CHANNELS)
                        w = pad_or_trim(w, self.n_samples)
                        by_label[lbl].append(w)
                    except Exception:
                        pass
                    break

        # Add donor clips for the swapped label
        if swap_label and swap_label in donor_clips:
            for clip in donor_clips[swap_label]:
                fname = Path(clip["path"]).name
                for ctype in ("source", "transient", "shared"):
                    subdir = CLIP_SUBDIRS.get(ctype, ctype)
                    p = self.data_root / subdir / fname
                    if p.exists():
                        try:
                            w = load_wav(str(p), self.sr, self.CHANNELS)
                            w = pad_or_trim(w, self.n_samples)
                            by_label[swap_label].append(w)
                        except Exception:
                            pass
                        break

        # Build 8 source channels — one per SEP_LABELS category
        channels = [self._sum_clips(by_label[lbl]) for lbl in SEP_LABELS]  # each (1, T)

        # Mixture = sum of all active channels
        mix_raw = sum(channels)
        peak    = mix_raw.abs().max().clamp(min=1e-8)
        mixture = mix_raw / peak
        channels_normed = [ch / peak for ch in channels]
        sources = torch.cat(channels_normed, dim=0)   # (8, T)

        # ── 2. Standard augmentation ──────────────────────────────────────────
        if self.augment:
            # Gain ±6 dB
            gain    = 10 ** (random.uniform(-6, 6) / 20)
            mixture = (mixture * gain).clamp(-1, 1)
            sources = (sources * gain).clamp(-1, 1)

            # Noise to mixture only
            noise_level = random.uniform(0.0, 0.02)
            mixture = (mixture + noise_level * torch.randn_like(mixture)).clamp(-1, 1)

            # Circular time shift
            if random.random() > 0.5:
                shift   = random.randint(0, self.sr // 2)
                mixture = torch.roll(mixture, shift)
                sources = torch.roll(sources, shift, dims=-1)

            # ── 3. SpecAugment on mixture ─────────────────────────────────────
            # Zero random frequency bands and time windows in the STFT domain
            # of the mixture only. Sources remain clean. Forces the model to
            # learn robust separation rather than memorising spectral patterns
            # of specific training recordings.
            if random.random() < self.SPEC_AUG_PROB:
                try:
                    mixture = spec_augment(mixture, self.sr)
                except Exception:
                    pass   # silent failure on edge cases (silent clips, etc.)

        return {
            "mixture":  mixture.squeeze(0),
            "sources":  sources,
            "label":    scene["label"],
            "scene_id": scene["id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataloader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_sep_dataloaders(
    labels_csv:    str,
    data_map_json: str,
    data_root:     str,
    audio_cfg:     AudioConfig = audio_cfg,
    batch_size:    int         = 4,
    num_workers:   int         = 0,
    val_split:     float       = 0.15,
    test_split:    float       = 0.10,
    seed:          int         = 42,
    repeat_factor: int         = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    gen = torch.Generator().manual_seed(seed)

    full_aug   = SoundSepDataset(labels_csv, data_map_json, data_root,
                                  audio_cfg, augment=True,
                                  repeat_factor=repeat_factor)
    full_noaug = SoundSepDataset(labels_csv, data_map_json, data_root,
                                  audio_cfg, augment=False,
                                  repeat_factor=1)

    # Split is based on real scene count (not repeated count)
    n_scenes = len(full_noaug)
    n_test   = int(n_scenes * test_split)
    n_val    = int(n_scenes * val_split)
    n_train  = n_scenes - n_val - n_test

    scene_indices = torch.randperm(n_scenes, generator=gen).tolist()
    train_scene_idx = scene_indices[:n_train]
    val_idx         = scene_indices[n_train:n_train + n_val]
    test_idx        = scene_indices[n_train + n_val:]

    # For training: expand each scene index into repeat_factor copies
    # e.g. scene 7 with repeat_factor=4 → indices 7, 7+n_scenes, 7+2*n_scenes, 7+3*n_scenes
    # All map to scene 7 via modulo, but each sees independent random augmentation
    train_idx = []
    for si in train_scene_idx:
        for r in range(repeat_factor):
            train_idx.append(si + r * n_scenes)

    train_ds = Subset(full_aug,   train_idx)
    val_ds   = Subset(full_noaug, val_idx)
    test_ds  = Subset(full_noaug, test_idx)

    print(f"[Splits]   train={n_train} scenes × {repeat_factor} = {len(train_ds)} steps  "
          f"val={len(val_ds)}  test={len(test_ds)}")
    print(f"[Remixing] ~{n_train**2:,} synthetic combinations available")
    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=(num_workers > 0),
        )

    return (make_loader(train_ds, True),
            make_loader(val_ds,   False),
            make_loader(test_ds,  False))