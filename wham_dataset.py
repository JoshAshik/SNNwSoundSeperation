"""
wham_dataset.py — pre-training dataset loader (MiniLibriMix backend).

Originally intended to use WHAM!, but wham_audio.zip has no free public
download — generating it requires the WSJ0 corpus (paid LDC licence) and the
Edinburgh DataShare handle 10283/3407 points to an unrelated chemistry paper.

We instead use MiniLibriMix (Zenodo record 3871592), which:
  - Is completely free and ~640 MB (vs 23 GB)
  - Has the identical separation task: 2-speaker clean mixtures
  - Uses the same folder layout: mix_clean/, s1/, s2/
  - Covers 800 train + 200 val mixtures at 8 kHz (resampled to 16 kHz)

Download instructions (run once):
    python wham_dataset.py --download --wham_root ./wham_data

Dataset structure after download:
    wham_data/
        MiniLibriMix/
            train/
                mix_clean/   ← two-speaker mixture
                s1/          ← speaker 1
                s2/          ← speaker 2
            val/
                mix_clean/
                s1/
                s2/

The public API (WHAMDataset, build_wham_dataloaders) is unchanged so the
rest of the pre-training pipeline requires no edits.

Split aliases accepted:
    "tr"  or "train" → MiniLibriMix/train/
    "cv"  or "val"   → MiniLibriMix/val/
    "tt"             → MiniLibriMix/val/  (MiniLibriMix has no separate test)
"""

import argparse
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset


TARGET_SR  = 16_000
N_SAMPLES  = TARGET_SR * 4   # 4 seconds

# Zenodo direct download — stable, no auth required
MINILIBRIMIX_URL = (
    "https://zenodo.org/record/3871592/files/MiniLibriMix.zip"
)

# Folder inside wham_root where MiniLibriMix lives
_MIX_ROOT = "MiniLibriMix"

# Map legacy WHAM! split names → MiniLibriMix folder names
_SPLIT_MAP = {
    "tr":    "train",
    "train": "train",
    "cv":    "val",
    "val":   "val",
    "tt":    "val",   # no separate test split in MiniLibriMix
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load wav → (1, T) mono, resampled to target_sr."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = T.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav


def pad_or_trim(wav: torch.Tensor, n: int) -> torch.Tensor:
    L = wav.shape[-1]
    if L >= n:
        return wav[..., :n]
    return F.pad(wav, (0, n - L))


def peak_norm(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return wav / wav.abs().max().clamp(min=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WHAMDataset(Dataset):
    """
    Two-speaker separation dataset backed by MiniLibriMix.

    Each item:
        mixture:  (T,)   — mix_clean, peak-normalised
        sources:  (2, T) — [s1, s2], same scale as mixture
        scene_id: str

    Augmentation (train only):
        - Random gain ±6 dB
        - Noise injection 0–1% to mixture
        - Random circular shift up to 1 s
        - Random source swap (s1↔s2)

    split: 'tr'/'train' (train), 'cv'/'val' (val), 'tt' → val fallback
    """

    def __init__(
        self,
        wham_root:   str,
        split:       str  = "tr",
        augment:     bool = False,
        max_samples: Optional[int] = None,
    ):
        self.wham_root = Path(wham_root)
        self.split     = split
        self.augment   = augment

        folder = _SPLIT_MAP.get(split, "train")
        split_dir = self.wham_root / _MIX_ROOT / folder
        mix_dir   = split_dir / "mix_clean"
        s1_dir    = split_dir / "s1"
        s2_dir    = split_dir / "s2"

        if not mix_dir.exists():
            raise FileNotFoundError(
                f"MiniLibriMix data not found at {mix_dir}\n"
                f"Run: python wham_dataset.py --download --wham_root {wham_root}"
            )

        mix_files = sorted(mix_dir.glob("*.wav"))
        if max_samples:
            mix_files = mix_files[:max_samples]

        self.samples = []
        for mix_f in mix_files:
            s1_f = s1_dir / mix_f.name
            s2_f = s2_dir / mix_f.name
            if s1_f.exists() and s2_f.exists():
                self.samples.append((str(mix_f), str(s1_f), str(s2_f)))

        print(f"[MiniLibriMix] split={split}({folder})  "
              f"samples={len(self.samples)}  augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        mix_path, s1_path, s2_path = self.samples[idx]

        mix = load_wav_mono(mix_path)
        s1  = load_wav_mono(s1_path)
        s2  = load_wav_mono(s2_path)

        mix = pad_or_trim(mix, N_SAMPLES)
        s1  = pad_or_trim(s1,  N_SAMPLES)
        s2  = pad_or_trim(s2,  N_SAMPLES)

        peak    = mix.abs().max().clamp(min=1e-8)
        mix     = mix / peak
        s1      = s1  / peak
        s2      = s2  / peak

        sources = torch.cat([s1, s2], dim=0)   # (2, T)

        if self.augment:
            gain = 10 ** (random.uniform(-6, 6) / 20)
            mix  = (mix * gain).clamp(-1, 1)
            sources = (sources * gain).clamp(-1, 1)

            noise_lvl = random.uniform(0.0, 0.01)
            mix = (mix + noise_lvl * torch.randn_like(mix)).clamp(-1, 1)

            if random.random() > 0.5:
                shift   = random.randint(0, TARGET_SR)
                mix     = torch.roll(mix, shift)
                sources = torch.roll(sources, shift, dims=-1)

            if random.random() > 0.5:
                sources = sources.flip(0)

        return {
            "mixture":  mix.squeeze(0),
            "sources":  sources,
            "scene_id": Path(mix_path).stem,
        }


def build_wham_dataloaders(
    wham_root:   str,
    batch_size:  int = 8,
    num_workers: int = 0,
    max_train:   Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = WHAMDataset(wham_root, split="tr", augment=True,
                           max_samples=max_train)
    val_ds   = WHAMDataset(wham_root, split="cv", augment=False)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=(num_workers > 0),
            drop_last=True,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_zip(path: Path) -> bool:
    """Return True if the file starts with the ZIP magic bytes PK\\x03\\x04."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def download_wham(wham_root: str):
    """
    Download MiniLibriMix (~640 MB) from Zenodo and unpack it.

    The original WHAM! wham_audio.zip has no free public download URL;
    generating it requires the paid WSJ0 corpus. MiniLibriMix is a
    freely available drop-in replacement with the same separation task.
    """
    root = Path(wham_root)
    root.mkdir(parents=True, exist_ok=True)

    already_there = root / _MIX_ROOT / "train" / "mix_clean"
    if already_there.exists():
        print(f"[download] MiniLibriMix already present at {root / _MIX_ROOT} "
              "— skipping download.")
        return

    zip_path = root / "MiniLibriMix.zip"
    url      = MINILIBRIMIX_URL

    print(f"[download] Downloading MiniLibriMix (~640 MB) from Zenodo ...")
    print(f"[download] URL: {url}\n")

    success = False

    # ── 1. Python requests ─────────────────────────────────────────────────
    if not success:
        try:
            import requests
            print("[download] Trying Python requests ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r  {pct:.1f}%  ({downloaded >> 20} MB)", end="")
            print()
            if _is_valid_zip(zip_path):
                success = True
                print("[download] requests download OK.")
            else:
                print("[download] requests: not a valid zip, discarding.")
                zip_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"\n[download] requests failed: {e}")
            zip_path.unlink(missing_ok=True)

    # ── 2. wget / curl ─────────────────────────────────────────────────────
    if not success:
        for tool, cmd in [
            ("wget", ["wget", "-O", str(zip_path), url]),
            ("curl", ["curl", "-L", "-o", str(zip_path), url]),
        ]:
            try:
                print(f"[download] Trying {tool} ...")
                subprocess.run(cmd, check=True)
                if _is_valid_zip(zip_path):
                    success = True
                    print(f"[download] {tool} download OK.")
                    break
                else:
                    print(f"[download] {tool}: not a valid zip, discarding.")
                    zip_path.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"[download] {tool} failed: {e}")
                zip_path.unlink(missing_ok=True)

    # ── 3. Manual fallback ─────────────────────────────────────────────────
    if not success:
        print("\n[download] Automatic download failed.")
        print("Download MiniLibriMix manually:\n")
        print(f"  1. Open: {url}")
        print(f"  2. Save as: {zip_path}")
        print(f"  3. Re-run: python wham_dataset.py --wham_root {wham_root}")
        sys.exit(1)

    print(f"\n[download] Unpacking {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    zip_path.unlink()
    print(f"[download] Done. Data at {root / _MIX_ROOT}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download",  action="store_true")
    p.add_argument("--wham_root", default="./wham_data")
    p.add_argument("--verify",    action="store_true",
                   help="Check dataset is intact after download")
    args = p.parse_args()

    if args.download:
        download_wham(args.wham_root)

    if args.verify or not args.download:
        print("\n[verify] Checking MiniLibriMix dataset ...")
        try:
            ds = WHAMDataset(args.wham_root, split="tr", augment=False)
            item = ds[0]
            print(f"  mixture shape: {item['mixture'].shape}")
            print(f"  sources shape: {item['sources'].shape}")
            print(f"  mixture max:   {item['mixture'].abs().max():.3f}")
            print(f"  Total train:   {len(ds)}")
            val_ds = WHAMDataset(args.wham_root, split="cv", augment=False)
            print(f"  Total val:     {len(val_ds)}")
            print("  OK")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
