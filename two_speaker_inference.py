"""
two_speaker_inference.py — CLI inference for a trained 2-speaker SNN-TasNet checkpoint.

Separates a mono mixture into two waveforms and optionally scores them with
SI-SDRi when reference sources are provided.

Usage:
    # Basic separation:
    python3 two_speaker_inference.py --model checkpoint.pt --input mix.wav

    # With references for SI-SDRi scoring:
    python3 two_speaker_inference.py --model checkpoint.pt --input mix.wav \\
        --ref1 source1.wav --ref2 source2.wav

    # Custom output dir, save input mixture copy:
    python3 two_speaker_inference.py --model checkpoint.pt --input mix.wav \\
        --out_dir ./results --save_mixture

Outputs (in --out_dir):
    speaker_1_est.wav   — peak-normalised separated channel 0
    speaker_2_est.wav   — peak-normalised separated channel 1
    mixture.wav         — copy of the input  (only with --save_mixture)

Architecture and EMA weights are loaded directly from the checkpoint — no
hyperparameters need to be specified manually.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from sep_model import SNNTasNet

SAMPLE_RATE = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> SNNTasNet:
    """
    Load SNNTasNet from a checkpoint.

    1. Reads model_cfg saved at training time — no manual arch flags needed.
    2. Loads EMA weights (ema_state["shadow"]) if present; falls back to raw
       model_state. Always prefer EMA for inference — smoother, more stable.
    3. Warns if the checkpoint was not trained with n_speakers=2.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── architecture ─────────────────────────────────────────────────────────
    model_cfg = ckpt.get("model_cfg", {})
    # Provide safe defaults that match the current DEFAULTS in snn_finetune.py
    defaults = dict(n_filters=256, kernel_sz=32, stride=16,
                    hidden=256, n_layers=3, dropout=0.3,
                    snn_mode="snn", snn_chunk=100, use_weight_norm=False)
    defaults.update(model_cfg)

    n_speakers = defaults.pop("n_speakers", 2)
    if n_speakers != 2:
        print(f"[warn] Checkpoint has n_speakers={n_speakers}. "
              "This script is intended for 2-speaker checkpoints. "
              "Proceeding — first two output channels will be saved.")

    model = SNNTasNet(**defaults, n_speakers=n_speakers).to(device)

    # ── weights ───────────────────────────────────────────────────────────────
    if "ema_state" in ckpt and ckpt["ema_state"].get("shadow"):
        shadow = ckpt["ema_state"]["shadow"]
        missing, unexpected = [], []
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
            else:
                missing.append(name)
        if missing:
            print(f"[warn] {len(missing)} params not in EMA shadow; "
                  "those weights are randomly initialised.")
        weight_src = "EMA weights"
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)
        weight_src = "raw model_state"

    model.eval()

    epoch  = ckpt.get("epoch", "?")
    val_si = ckpt.get("val_si_sdri", float("nan"))
    print(
        f"[model] loaded  ckpt={Path(ckpt_path).name}  "
        f"epoch={epoch}  val_SI-SDRi={val_si:+.2f} dB  "
        f"n_speakers={n_speakers}  "
        f"snn_mode={defaults.get('snn_mode','?')}  "
        f"weights={weight_src}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, max_seconds: float = 30.0) -> torch.Tensor:
    """
    Load a WAV file → mono float32 tensor of shape (1, T) at SAMPLE_RATE.

    - Stereo → mono by averaging channels.
    - Resamples if the source rate differs from SAMPLE_RATE.
    - Truncates to max_seconds (default 30 s) to avoid OOM on long files.

    Returns: (1, T) tensor, values in [-1, 1].
    """
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)   # stereo → mono
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
        print(f"[audio] resampled {sr} Hz → {SAMPLE_RATE} Hz")

    max_samples = int(max_seconds * SAMPLE_RATE)
    if wav.shape[-1] > max_samples:
        print(f"[audio] truncated to {max_seconds:.1f} s "
              f"({wav.shape[-1] / SAMPLE_RATE:.1f} s input). "
              "Use --max_seconds to increase the limit.")
        wav = wav[..., :max_samples]

    dur = wav.shape[-1] / SAMPLE_RATE
    print(f"[audio] loaded   {Path(path).name}  "
          f"{dur:.2f} s  ({wav.shape[-1]} samples)")
    return wav.float()   # (1, T)


def peak_norm(wav: torch.Tensor) -> torch.Tensor:
    """Divide by peak absolute value. Safe against silent inputs."""
    peak = wav.abs().max().clamp(min=1e-8)
    return wav / peak


def save_wav(path: str, wav: torch.Tensor) -> None:
    """Peak-normalise, hard-clamp to [-1, 1], and save as 16kHz mono WAV."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = peak_norm(wav).clamp(-1.0, 1.0).cpu()
    torchaudio.save(path, wav, SAMPLE_RATE)
    rms_db = 10 * torch.log10(wav.pow(2).mean().clamp(min=1e-10))
    print(f"  saved  {Path(path).name}  "
          f"RMS={rms_db.item():.1f} dBFS  "
          f"peak={wav.abs().max().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Separation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def separate(
    model: SNNTasNet,
    mixture: torch.Tensor,
    device: torch.device,
    input_gain: float = 1.0,
) -> dict:
    """
    Run a (1, T) mixture through the model.

    Pipeline (matches SNNTasNet.forward internally):
      1. Encoder:   Conv1d → (1, n_filters, L)
      2. Separator: LIF chunks → sigmoid masks → (1, n_speakers, n_filters, L)
      3. Decoder:   mask[c] * encoded → ConvTranspose1d → (1, n_speakers, T')
         where T' may be slightly larger than T due to ConvTranspose1d padding.
         We trim outputs back to T to match the input length.

    Returns a dict:
        "waveforms"  : list of (1, T) tensors, one per speaker
        "masks"      : (1, n_speakers, n_filters, L) float tensor
        "spike_recs" : list of per-layer spike records (from snntorch)
    """
    T_in  = mixture.shape[-1]

    # Reset snntorch membrane state before every forward pass.
    # snn.Leaky stores self.mem internally even when explicit mem is passed to
    # forward(). Under certain CUDA scheduling orders that stored state persists
    # across calls and bleeds into the next run — causing the non-deterministic
    # reconstruction error explosions (run 1: 5.13, run 2: 58.58).
    try:
        import snntorch as _snn
        _snn.utils.reset(model)
    except Exception:
        pass

    # RMS-normalise to unit energy — matches v6 training where each source was
    # normalised to RMS=1.0 before mixing. Peak-normalising a mixture whose
    # sources have arbitrary recording levels puts the encoder input at an
    # unpredictable scale, over-driving LIF neurons.
    # Hard 0.1 pre-scale brings drive down to the range the LIF neurons were
    # calibrated for (target spike rate ~0.20; currently 43% without this).
    # input_gain lets you fine-tune further from the command line.
    rms   = mixture.pow(2).mean().clamp(min=1e-8).sqrt()
    mix_n = (mixture / rms * 0.1 * input_gain).to(device)  # (1, T)

    out   = model(mix_n)          # separated: (1, C, T'), masks: (1, C, F, L)
    sep   = out["separated"]      # (1, C, T')

    # Trim ConvTranspose1d overshoot to match input length exactly
    T_out = sep.shape[-1]
    if T_out > T_in:
        sep = sep[..., :T_in]
    elif T_out < T_in:
        sep = F.pad(sep, (0, T_in - T_out))

    waveforms = [sep[0, c:c+1].cpu() for c in range(sep.shape[1])]

    return {
        "waveforms":  waveforms,           # list of (1, T) on CPU
        "masks":      out["masks"].cpu(),  # (1, C, F, L)
        "spike_recs": out.get("spike_recs", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Scale-invariant SDR between 1-D tensors.
    Projects estimate onto reference; residual is the noise.
    """
    eps = 1e-8
    ref  = reference - reference.mean()
    est  = estimate  - estimate.mean()
    dot  = (est * ref).sum()
    ref2 = (ref * ref).sum().clamp(min=eps)
    proj = (dot / ref2) * ref          # projection of est onto ref
    noise = est - proj
    ratio = proj.pow(2).sum() / noise.pow(2).sum().clamp(min=eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps)).item()


def si_sdri(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mixture: torch.Tensor,
) -> float:
    """
    SI-SDR improvement: SI-SDR(estimate, reference) − SI-SDR(mixture, reference).
    All inputs: 1-D tensors of equal length.
    """
    return _si_sdr(estimate.flatten(), reference.flatten()) \
         - _si_sdr(mixture.flatten(),  reference.flatten())


def print_diagnostics(
    result:  dict,
    mixture: torch.Tensor,
    refs:    list,            # list of (1, T) tensors or None; len == n_speakers
) -> None:
    """Print spike rates, mask statistics, and optional SI-SDRi per channel."""

    masks      = result["masks"]          # (1, C, F, L)
    spike_recs = result["spike_recs"]
    C          = len(result["waveforms"])

    print("\n── Diagnostics ─────────────────────────────────────────────────────")

    # ── waveform energy ───────────────────────────────────────────────────────
    mix_rms = mixture.pow(2).mean().sqrt().item()
    print(f"Input RMS:  {mix_rms:.4f}")
    for i, wav in enumerate(result["waveforms"]):
        rms = wav.pow(2).mean().sqrt().item()
        print(f"  ch{i+1} output RMS: {rms:.4f}  "
              f"({'active' if rms > 1e-4 else 'silent'})")

    # ── SI-SDRi (only if references were provided) ────────────────────────────
    if any(r is not None for r in refs):
        print("\nSI-SDRi vs references:")
        mix_1d = mixture.flatten()
        # Try both permutations and pick best assignment (2-speaker PIT)
        wavs   = result["waveforms"]
        perm_scores = {}
        if refs[0] is not None and refs[1] is not None:
            r0, r1 = refs[0].flatten(), refs[1].flatten()
            # Trim to common length
            L = min(mix_1d.shape[0], r0.shape[0], r1.shape[0],
                    wavs[0].flatten().shape[0])
            s0 = wavs[0].flatten()[:L]
            s1 = wavs[1].flatten()[:L] if len(wavs) > 1 else torch.zeros(L)
            mix_trim = mix_1d[:L]

            score_01 = si_sdri(s0, r0[:L], mix_trim) + si_sdri(s1, r1[:L], mix_trim)
            score_10 = si_sdri(s0, r1[:L], mix_trim) + si_sdri(s1, r0[:L], mix_trim)

            if score_01 >= score_10:
                best = [(s0, r0[:L]), (s1, r1[:L])]
                perm_label = "ch1→ref1, ch2→ref2"
            else:
                best = [(s0, r1[:L]), (s1, r0[:L])]
                perm_label = "ch1→ref2, ch2→ref1"

            print(f"  Best permutation: {perm_label}")
            for i, (est, ref) in enumerate(best):
                score = si_sdri(est, ref, mix_trim)
                print(f"  Speaker {i+1}: SI-SDRi = {score:+.2f} dB")
        else:
            # Single reference provided — score each channel independently
            for i, (wav, ref) in enumerate(zip(wavs, refs)):
                if ref is not None:
                    L    = min(mix_1d.shape[0], ref.shape[0],
                               wav.flatten().shape[0])
                    score = si_sdri(wav.flatten()[:L], ref.flatten()[:L],
                                    mix_1d[:L])
                    print(f"  Speaker {i+1} vs ref{i+1}: SI-SDRi = {score:+.2f} dB")

    # ── spike rates ───────────────────────────────────────────────────────────
    if spike_recs:
        print("\nSpike rates (target ~0.10):")
        for i, spk in enumerate(spike_recs):
            if isinstance(spk, torch.Tensor):
                rate = spk.float().mean().item()
                flag = "  ← SATURATED" if rate > 0.25 else ""
                print(f"  Layer {i+1}: {rate:.4f}{flag}")

    # ── mask statistics ───────────────────────────────────────────────────────
    if masks is not None and masks.numel() > 0:
        print("\nMask statistics (per channel):")
        for c in range(min(C, masks.shape[1])):
            m = masks[0, c]                              # (F, L)
            sat  = ((m > 0.9) | (m < 0.1)).float().mean().item()
            mean = m.mean().item()
            print(f"  ch{c+1}: mean={mean:.3f}  saturation={sat*100:.1f}%  "
                  f"({'binary — model decided' if sat > 0.5 else 'soft — still uncertain'})")

    # ── reconstruction check ──────────────────────────────────────────────────
    total_out = sum(w for w in result["waveforms"])
    L_chk = min(total_out.shape[-1], mixture.shape[-1])
    recon_err = (total_out.flatten()[:L_chk]
                 - mixture.flatten()[:L_chk]).abs().mean().item()
    mix_scale = mixture.abs().mean().clamp(min=1e-8).item()
    print(f"\nReconstruction error (rel L1): {recon_err / mix_scale:.4f}  "
          f"(target < 0.10; high = energy not conserved)")
    print("────────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-speaker SNN-TasNet inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--model", required=True,
        help="Path to trained checkpoint (.pt)\n"
             "e.g. checkpoints_snn/best_snn_v2.pt",
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the input mixture WAV file",
    )
    p.add_argument(
        "--out_dir", default="./pit_output",
        help="Directory for output WAV files (default: ./pit_output)",
    )
    p.add_argument(
        "--ref1", default=None,
        help="(Optional) Reference WAV for speaker 1 — enables SI-SDRi scoring",
    )
    p.add_argument(
        "--ref2", default=None,
        help="(Optional) Reference WAV for speaker 2 — enables SI-SDRi scoring",
    )
    p.add_argument(
        "--save_mixture", action="store_true",
        help="Also save the input mixture to out_dir as mixture.wav",
    )
    p.add_argument(
        "--device", default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto — uses CUDA if available)",
    )
    p.add_argument(
        "--max_seconds", type=float, default=30.0,
        help="Truncate input to this many seconds (default: 30.0) to avoid OOM",
    )
    p.add_argument(
        "--input_gain", type=float, default=1.0,
        help="Multiplier on top of the hard 0.1 pre-scale and RMS normalisation.\n"
             "Effective input level = (mixture / rms) * 0.1 * input_gain.\n"
             "Try 2.0 if separated audio is too quiet; 0.5 if spike rates still > 0.30.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    model = load_model(args.model, device)

    # ── load input ────────────────────────────────────────────────────────────
    mixture = load_audio(args.input, max_seconds=args.max_seconds)

    # ── load optional references ──────────────────────────────────────────────
    refs = [None, None]
    for i, ref_path in enumerate([args.ref1, args.ref2]):
        if ref_path is not None:
            refs[i] = load_audio(ref_path, max_seconds=args.max_seconds)
            # Trim reference to match mixture length
            L = mixture.shape[-1]
            if refs[i].shape[-1] > L:
                refs[i] = refs[i][..., :L]
            elif refs[i].shape[-1] < L:
                refs[i] = F.pad(refs[i], (0, L - refs[i].shape[-1]))

    # v6: RMS-normalise references to unit energy — matches the v6 training
    # dataset which normalises both sources to RMS=1.0 before mixing.
    # Ensures SI-SDRi is evaluated against sources at a consistent scale
    # regardless of original clip loudness.
    for i in range(len(refs)):
        if refs[i] is not None:
            rms = refs[i].pow(2).mean().clamp(min=1e-8).sqrt()
            refs[i] = refs[i] / rms

    # ── separate ──────────────────────────────────────────────────────────────
    print("\n[inference] running forward pass...")
    result = separate(model, mixture, device, input_gain=args.input_gain)
    n_out  = len(result["waveforms"])
    print(f"[inference] done — {n_out} output channel(s)")

    # ── save outputs ──────────────────────────────────────────────────────────
    print(f"\n[saving] → {out_dir}/")
    for i, wav in enumerate(result["waveforms"]):
        out_path = str(out_dir / f"speaker_{i+1}_est.wav")
        save_wav(out_path, wav)

    if args.save_mixture:
        save_wav(str(out_dir / "mixture.wav"), mixture)

    # ── diagnostics ───────────────────────────────────────────────────────────
    print_diagnostics(result, mixture, refs)

    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
