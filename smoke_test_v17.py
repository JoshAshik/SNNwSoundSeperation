"""
smoke_test_v17.py — end-to-end smoke test for the v17 pipeline.

Generates a tiny synthetic Libri2Mix-shaped dataset (train-100/ + dev/, each with
mix_clean/ s1/ s2/, where mix = s1 + s2 exactly by construction), then exercises:
  1. two_speaker_train_v17.py  — 2 epochs, tiny batch, short clips (full dataloader
     + finer-encoder forward/backward + plateau scheduler + checkpoint save path).
  2. oracle_mask_diagnostic.py — STFT oracle masks on the synthetic dev split.

This does NOT validate separation quality (random signals are unseparable) — it
validates that every code path runs to exit 0 and produces the expected artifacts
before deploying to ARC. Everything is written to a temp dir and cleaned up.

Usage:
    python3 smoke_test_v17.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio

SR        = 16_000
CLIP_LEN  = 8_000           # 0.5 s — keep the smoke test fast
N_TRAIN   = 6
N_DEV     = 4


def _synth_source(seed: int, n: int) -> torch.Tensor:
    """A band-limited-ish synthetic source: a few sinusoids + low noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / SR
    sig = torch.zeros(n)
    for _ in range(3):
        f     = float(torch.randint(120, 3000, (1,), generator=g))
        phase = float(torch.rand(1, generator=g)) * 2 * math.pi
        amp   = 0.2 + 0.3 * float(torch.rand(1, generator=g))
        sig  += amp * torch.sin(2 * math.pi * f * t + phase)
    sig += 0.01 * torch.randn(n, generator=g)
    return sig


def _make_split(root: Path, split: str, n_items: int) -> None:
    for sub in ("mix_clean", "s1", "s2"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n_items):
        # Slightly varying lengths so crop/pad logic gets exercised too.
        n = CLIP_LEN + (i % 3) * 1000
        s1 = _synth_source(1000 * i + 1, n)
        s2 = _synth_source(1000 * i + 2, n)
        mix = s1 + s2
        # Peak-normalise the mixture (mirrors how real wavs sit in [-1, 1]).
        peak = mix.abs().max().clamp(min=1e-8)
        s1, s2, mix = s1 / peak, s2 / peak, mix / peak
        stem = f"utt{i:03d}"
        torchaudio.save(str(root / split / "mix_clean" / f"{stem}.wav"),
                        mix.unsqueeze(0), SR)
        torchaudio.save(str(root / split / "s1" / f"{stem}.wav"),
                        s1.unsqueeze(0), SR)
        torchaudio.save(str(root / split / "s2" / f"{stem}.wav"),
                        s2.unsqueeze(0), SR)


def _run(cmd: list, cwd: Path) -> int:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def main() -> int:
    repo = Path(__file__).resolve().parent
    tmp  = Path(tempfile.mkdtemp(prefix="v17_smoke_"))
    print(f"[smoke] repo  = {repo}")
    print(f"[smoke] tmp   = {tmp}")

    failures = []
    try:
        data = tmp / "librimix"
        print(f"\n[smoke] Generating synthetic Libri2Mix at {data}")
        _make_split(data, "train-100", N_TRAIN)
        _make_split(data, "dev",       N_DEV)

        # Verify the mix = s1 + s2 invariant on one sample.
        m, _ = torchaudio.load(str(data / "dev/mix_clean/utt000.wav"))
        a, _ = torchaudio.load(str(data / "dev/s1/utt000.wav"))
        b, _ = torchaudio.load(str(data / "dev/s2/utt000.wav"))
        err = (m - a - b).abs().mean().item()
        print(f"[smoke] mix=s1+s2 invariant: mean|err|={err:.2e} (should be ~0)")

        py = sys.executable

        # ── 1. Training smoke (2 epochs) ──────────────────────────────────────
        print("\n" + "=" * 70)
        print("[smoke] 1/3  Training (2 epochs, finer encoder, from scratch)")
        print("=" * 70)
        rc = _run([
            py, "two_speaker_train_v17.py",
            "--librimix_root", str(data),
            "--train_split", "train-100", "--val_split", "dev",
            "--clip_len", str(CLIP_LEN),
            "--n_epochs", "2", "--batch_size", "2", "--num_workers", "0",
            "--val_every", "1", "--max_wall_hours", "0",
            "--ckpt_dir", str(tmp / "ckpt"),
            "--log_dir",  str(tmp / "runs"),
            "--csv_path", str(tmp / "v17_smoke_log.csv"),
        ], cwd=repo)
        if rc != 0:
            failures.append(f"training exited {rc}")
        for artifact in ("ckpt/best_2spk.pt", "ckpt/best_2spk_latest.pt",
                         "v17_smoke_log.csv"):
            if (tmp / artifact).exists():
                print(f"[smoke]   artifact OK: {artifact}")
            else:
                failures.append(f"missing artifact {artifact}")

        # ── 2. Resume smoke (continue 2 -> 3 epochs) ──────────────────────────
        print("\n" + "=" * 70)
        print("[smoke] 2/3  Resume (load latest, train 1 more epoch)")
        print("=" * 70)
        rc = _run([
            py, "two_speaker_train_v17.py", "--resume",
            "--librimix_root", str(data),
            "--train_split", "train-100", "--val_split", "dev",
            "--clip_len", str(CLIP_LEN),
            "--n_epochs", "3", "--batch_size", "2", "--num_workers", "0",
            "--val_every", "1", "--max_wall_hours", "0",
            "--ckpt_dir", str(tmp / "ckpt"),
            "--log_dir",  str(tmp / "runs"),
            "--csv_path", str(tmp / "v17_smoke_log.csv"),
        ], cwd=repo)
        if rc != 0:
            failures.append(f"resume exited {rc}")
        else:
            ck = torch.load(tmp / "ckpt/best_2spk_latest.pt",
                            map_location="cpu", weights_only=False)
            ep = ck.get("epoch", 0)
            print(f"[smoke]   resumed checkpoint epoch={ep} (expect 3)")
            if ep != 3:
                failures.append(f"resume reached epoch {ep}, expected 3")
            cfg = ck.get("model_cfg", {})
            if (cfg.get("kernel_sz"), cfg.get("stride")) != (16, 8):
                failures.append(f"ckpt encoder cfg {cfg.get('kernel_sz')}/"
                                f"{cfg.get('stride')} != 16/8")

        # ── 3. Oracle diagnostic smoke ────────────────────────────────────────
        print("\n" + "=" * 70)
        print("[smoke] 3/3  Oracle mask diagnostic")
        print("=" * 70)
        rc = _run([
            py, "oracle_mask_diagnostic.py",
            "--librimix_root", str(data),
            "--split", "dev", "--max_samples", str(N_DEV),
            "--windows", "32:16", "16:8",
            "--output_csv", str(tmp / "oracle_smoke.csv"),
        ], cwd=repo)
        if rc != 0:
            failures.append(f"oracle exited {rc}")
        elif (tmp / "oracle_smoke.csv").exists():
            print("[smoke]   artifact OK: oracle_smoke.csv")
        else:
            failures.append("missing artifact oracle_smoke.csv")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n[smoke] cleaned up {tmp}")

    print("\n" + "=" * 70)
    if failures:
        print("[smoke] FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("[smoke] ALL PASSED — v17 train/resume/oracle paths run to exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
