"""
inference.py – Run the trained SNN separator on a new audio file.

Usage:
    python inference.py --audio path/to/mix.wav --checkpoint checkpoints/best_model.pt

Output:
    - Detected source labels with confidence scores
    - Per-slot embeddings (for downstream source-specific processing)
    - Optional: save per-slot reconstructed audio (requires a decoder, see note)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

from config import ALL_LABELS, SOURCE_LABELS, TRANSIENT_LABELS, audio_cfg, cochlea_cfg, enc_cfg, snn_cfg
from dataset import load_wav, pad_or_trim
from encoding import AudioToSpikes
from model import SNNSoundSeparator


# ──────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt    = torch.load(checkpoint_path, map_location=device)
    encoder = AudioToSpikes(cochlea_cfg, audio_cfg, enc_cfg).to(device)
    model   = SNNSoundSeparator(snn_cfg).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    model.load_state_dict(ckpt["model_state"])
    encoder.eval()
    model.eval()
    print(f"[inference] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val F1={ckpt.get('val_f1', '?'):.3f}")
    return encoder, model


# ──────────────────────────────────────────────────────────────────────────────
# Single-file inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    audio_path:    str,
    encoder:       AudioToSpikes,
    model:         SNNSoundSeparator,
    device:        torch.device,
    threshold:     float = 0.5,
) -> Dict:
    """
    Run the model on one audio file.

    Returns a dict with:
        labels:      list of detected class names
        scores:      dict {label: confidence}
        slot_embeds: (n_slots, sep_dim) numpy array
    """
    waveform = load_wav(audio_path, target_sr=audio_cfg.sample_rate,
                        target_channels=audio_cfg.channels)
    waveform = pad_or_trim(waveform, audio_cfg.n_samples)
    waveform = waveform.unsqueeze(0).to(device)                 # (1, 2, T)

    with torch.no_grad():
        spikes = encoder(waveform)                              # (T, 1, D)
        out    = model(spikes)

    probs  = out["logits"].sigmoid().squeeze(0).cpu()           # (8,)
    slots  = out["slot_embeddings"].squeeze(0).cpu().numpy()    # (n_slots, D)

    scores    = {lbl: probs[i].item() for i, lbl in enumerate(ALL_LABELS)}
    det_labels = [lbl for lbl, sc in scores.items() if sc >= threshold]

    return {
        "detected_labels": det_labels,
        "scores":          scores,
        "slot_embeddings": slots,
    }


def pretty_print(result: Dict, audio_path: str):
    print(f"\n{'─'*50}")
    print(f" File: {audio_path}")
    print(f"{'─'*50}")
    print(" Detected labels:")
    for lbl in result["detected_labels"]:
        print(f"   ✓ {lbl:<30}  {result['scores'][lbl]:.3f}")
    print("\n All label scores:")
    for lbl, sc in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(sc * 20)
        print(f"   {lbl:<30}  {sc:.3f}  {bar}")
    print(f"\n Slot embeddings shape: {result['slot_embeddings'].shape}")
    print(f"{'─'*50}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",      required=True,  help="Path to input WAV file")
    parser.add_argument("--checkpoint", required=True,  help="Path to model checkpoint")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, model = load_checkpoint(args.checkpoint, device)

    result = predict(args.audio, encoder, model, device, threshold=args.threshold)
    pretty_print(result, args.audio)
