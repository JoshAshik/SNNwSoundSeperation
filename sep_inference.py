"""
sep_inference.py — Load best checkpoint and generate separated audio.

Usage:
    # Separate a single wav file:
    python sep_inference.py --input your_mixture.wav

    # Pull random scenes from the training dataset:
    python sep_inference.py --from_dataset --n_samples 5 --data_root ./data

Outputs saved to ./sep_output/:
    {name}_mixture.wav              — original mix
    {name}_{label}.wav              — one file per active sound category
                                      (silent channels are skipped)

Channel → label mapping (ALL_LABELS order from config.py):
    0: human_voice
    1: animal
    2: impacts
    3: music
    4: weather
    5: alerts
    6: domestic_and_objects
    7: tools_and_machines

Model architecture, n_speakers, and EMA weights are loaded directly from the
checkpoint — no need to manually match any hyperparameters.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

from sep_model   import SNNTasNet
from sep_dataset import SEP_LABELS


SAMPLE_RATE       = 16_000
N_SAMPLES         = SAMPLE_RATE * 3   # 3 seconds = 48000 samples
SILENCE_THRESHOLD = 1e-3              # RMS below this → channel is silent / absent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> SNNTasNet:
    """
    Load model from checkpoint. Reads architecture config saved during training
    so all hyperparameters are restored automatically. Uses EMA weights if available.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = ckpt.get("model_cfg", {
        "n_filters":  256,
        "kernel_sz":  32,
        "stride":     16,
        "hidden":     192,
        "n_layers":   2,
        "dropout":    0.3,
        "snn_mode":   "snn",
        "snn_chunk":  100,
    })
    # n_speakers defaults to len(SEP_LABELS)=8 for new checkpoints;
    # falls back to 8 if old checkpoint doesn't have it
    n_speakers = model_cfg.pop("n_speakers", len(SEP_LABELS))

    model = SNNTasNet(**model_cfg, n_speakers=n_speakers).to(device)

    if "ema_state" in ckpt:
        print("[model] Using EMA weights")
        for name, param in model.named_parameters():
            if name in ckpt["ema_state"]["shadow"]:
                param.data.copy_(ckpt["ema_state"]["shadow"][name])
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    epoch  = ckpt.get("epoch", "?")
    val_si = ckpt.get("val_si_sdri", float("nan"))
    print(f"[model] epoch={epoch}  val SI-SDRi={val_si:+.2f} dB  "
          f"n_speakers={n_speakers}  mode={model_cfg.get('snn_mode','?')}")
    return model


def load_wav_mono(path: str) -> torch.Tensor:
    """Load any wav → (1, T) mono at 16kHz, trimmed/padded to 3s."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.mean(0, keepdim=True)
    if wav.shape[-1] >= N_SAMPLES:
        wav = wav[..., :N_SAMPLES]
    else:
        wav = F.pad(wav, (0, N_SAMPLES - wav.shape[-1]))
    return wav   # (1, T)


def peak_norm(wav: torch.Tensor) -> torch.Tensor:
    peak = wav.abs().max().clamp(min=1e-8)
    return wav / peak


def save_wav(path: str, wav: torch.Tensor):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(path, wav.clamp(-1.0, 1.0).cpu(), SAMPLE_RATE)
    print(f"  saved → {path}")


def is_active(wav: torch.Tensor, threshold: float = SILENCE_THRESHOLD) -> bool:
    """Return True if the waveform has meaningful energy (not silent)."""
    return wav.pow(2).mean().sqrt().item() > threshold


@torch.no_grad()
def separate(model: SNNTasNet, mixture: torch.Tensor,
             device: torch.device) -> dict:
    """
    Run inference on a (1, T) mixture.

    Returns a dict {label: (1, T) waveform} for each channel,
    including silent channels (caller decides whether to save them).
    """
    mix_norm = peak_norm(mixture).to(device)
    out      = model(mix_norm)                  # separated: (1, C, T)
    sep      = out["separated"][0]              # (C, T)
    C        = sep.shape[0]

    results = {}
    for c in range(C):
        label = SEP_LABELS[c] if c < len(SEP_LABELS) else f"source_{c}"
        wav   = sep[c:c+1].cpu()                # (1, T) — raw, not normalized
        results[label] = wav
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: single file
# ─────────────────────────────────────────────────────────────────────────────

def run_on_file(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.checkpoint, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[inference] Processing: {args.input}")
    mixture  = load_wav_mono(args.input)
    channels = separate(model, mixture, device)

    stem = Path(args.input).stem
    save_wav(str(out_dir / f"{stem}_mixture.wav"), peak_norm(mixture))

    active_labels = []
    for label, wav in channels.items():
        if is_active(wav):
            save_wav(str(out_dir / f"{stem}_{label}.wav"), peak_norm(wav))
            active_labels.append(label)
        else:
            print(f"  skip  -> {label} (silent)")

    print(f"\nActive categories detected: {active_labels}")
    print(f"Done. Files in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: pull scenes from dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_on_dataset(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.checkpoint, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    with open(args.data_map_json) as f:
        raw = json.load(f)

    scenes = raw["data"]
    random.shuffle(scenes)
    scenes = scenes[:args.n_samples]

    CLIP_SUBDIRS = {"source": "sources", "transient": "transients"}

    def find_clip(clip_dict, clip_type):
        fname  = Path(clip_dict["path"]).name
        subdir = CLIP_SUBDIRS.get(clip_type, "sources")
        for folder in (subdir, "shared"):
            p = data_root / folder / fname
            if p.exists():
                return str(p)
        return None

    def load_and_sum(clips, clip_type):
        wavs = []
        for c in clips:
            p = find_clip(c, clip_type)
            if p:
                try:
                    wavs.append(load_wav_mono(p))
                except Exception:
                    pass
        if not wavs:
            return torch.zeros(1, N_SAMPLES)
        normed = [peak_norm(w) for w in wavs]
        return torch.stack(normed).sum(0)

    for scene in scenes:
        sid = scene["id"]
        print(f"\n[scene {sid}]")

        # Show what categories are present according to the data map
        all_clips  = scene.get("source_clips", []) + scene.get("transient_clips", [])
        present    = sorted({c.get("unified_label", "?") for c in all_clips})
        print(f"  Labels in scene: {present}")

        src_wav = load_and_sum(scene.get("source_clips",    []), "source")
        trn_wav = load_and_sum(scene.get("transient_clips", []), "transient")
        mixture = peak_norm(src_wav + trn_wav)

        if mixture.abs().max() < 1e-6:
            print(f"  ⚠ no audio found, skipping")
            continue

        channels = separate(model, mixture, device)
        save_wav(str(out_dir / f"scene{sid}_mixture.wav"), mixture)

        active_labels = []
        for label, wav in channels.items():
            if is_active(wav):
                save_wav(str(out_dir / f"scene{sid}_{label}.wav"), peak_norm(wav))
                active_labels.append(label)
            else:
                print(f"  skip  -> {label} (silent)")

        print(f"  Active outputs: {active_labels}")

    print(f"\nDone. Files in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="./checkpoints_snn/best_snn.pt")
    p.add_argument("--output_dir",    default="./sep_output")
    p.add_argument("--input",         default=None,
                   help="Path to a wav file to separate")
    p.add_argument("--from_dataset",  action="store_true")
    p.add_argument("--data_root",     default="./data")
    p.add_argument("--data_map_json", default="./original_data_map.json")
    p.add_argument("--n_samples",     type=int, default=5)
    args = p.parse_args()

    if args.input:
        run_on_file(args)
    elif args.from_dataset:
        run_on_dataset(args)
    else:
        print("Specify --input <file.wav>  or  --from_dataset")
        print("  python sep_inference.py --input my_mix.wav")
        print("  python sep_inference.py --from_dataset --n_samples 5 --data_root ./data")
