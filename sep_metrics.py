"""
sep_metrics.py — SI-SDRi, PESQ for evaluation.

Direct (no-PIT) computation — channels map to fixed label slots (ALL_LABELS order).

SI-SDRi = mean over *active* channels of:
          SI-SDR(estimate[c], target[c]) - SI-SDR(mixture, target[c])
PESQ    = perceptual speech quality score 1.0–4.5  [via pesq]

pip install pesq
"""

from typing import Dict
import numpy as np
import torch

from sep_losses import si_sdr

try:
    from pesq import pesq as _pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("[metrics] pip install pesq  to enable PESQ")


def _active_channels(targets: torch.Tensor) -> torch.Tensor:
    """Return bool (C,) mask: True if any batch item has signal in channel c."""
    return (targets.abs().mean(dim=-1) >= 1e-4).any(dim=0)   # (C,)


def fast_si_sdri(estimates: torch.Tensor, targets: torch.Tensor,
                 mixture: torch.Tensor) -> float:
    """
    Quick SI-SDRi for training monitoring — direct per-channel, no PIT.
    Only averages over channels that have an active target in the batch.
    O(C) — safe for C=8.
    """
    active = _active_channels(targets)   # (C,)
    if not active.any():
        return 0.0
    C = estimates.shape[1]
    vals = []
    for c in range(C):
        if not active[c].item():
            continue
        sisdr_e = si_sdr(estimates[:, c], targets[:, c])   # (B,)
        sisdr_m = si_sdr(mixture,         targets[:, c])   # (B,)
        vals.append((sisdr_e - sisdr_m).mean().item())
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def compute_metrics(
    estimates:   torch.Tensor,   # (B, C, T)
    targets:     torch.Tensor,   # (B, C, T)
    mixture:     torch.Tensor,   # (B, T)
    sample_rate: int = 16_000,
    use_pesq:    bool = True,    # set False for non-speech datasets (FSD50K, etc.)
) -> Dict[str, float]:
    """Direct per-channel SI-SDRi and PESQ (no PIT). Safe for any C."""
    B, C, T = estimates.shape
    est = estimates.cpu().float()
    tgt = targets.cpu().float()
    mix = mixture.cpu().float()

    active = _active_channels(tgt)   # (C,) bool tensor

    si_sdri_list, pesq_list = [], []

    for b in range(B):
        ch_vals = []
        for c in range(C):
            if not active[c].item():
                continue
            sisdr_e = si_sdr(est[b, c:c+1], tgt[b, c:c+1]).item()
            sisdr_m = si_sdr(mix[b:b+1],    tgt[b, c:c+1]).item()
            ch_vals.append(sisdr_e - sisdr_m)
        if ch_vals:
            si_sdri_list.append(float(np.mean(ch_vals)))

        if HAS_PESQ and use_pesq:
            try:
                mode = "wb" if sample_rate == 16000 else "nb"
                scores = []
                for c in range(C):
                    if not active[c].item():
                        continue
                    ref = tgt[b, c].numpy()
                    deg = est[b, c].numpy()
                    # Skip near-silent channels — PESQ native code segfaults on them
                    if np.abs(ref).max() < 1e-4 or np.abs(deg).max() < 1e-4:
                        continue
                    ref = ref / (np.abs(ref).max() + 1e-8)
                    deg = deg / (np.abs(deg).max() + 1e-8)
                    # PESQ is only valid for speech — skip non-zero but NaN/Inf signals
                    if not (np.isfinite(ref).all() and np.isfinite(deg).all()):
                        continue
                    scores.append(_pesq(sample_rate, ref.astype(np.float32),
                                        deg.astype(np.float32), mode))
                if scores:
                    pesq_list.append(float(np.mean(scores)))
            except Exception:
                pass

    result = {"si_sdri": float(np.mean(si_sdri_list)) if si_sdri_list else 0.0}
    if pesq_list:
        result["pesq"] = float(np.mean(pesq_list))
    return result


@torch.no_grad()
def diagnostic_metrics(
    estimates:  torch.Tensor,               # (B, C, T)
    targets:    torch.Tensor,               # (B, C, T)
    mixture:    torch.Tensor,               # (B, T)
    masks:      "torch.Tensor | None" = None,   # (B, C, N, L) conv masks
    spike_recs: "list | None"         = None,   # list of spike tensors from SNN
) -> Dict[str, float]:
    """
    Diagnostic metrics — direct per-channel (no PIT).

    Returns a flat dict with:
      si_sdri_s{N}          — per-channel SI-SDRi (active channels only)
      reconstruction_l1_rel — |sum(est) - mix| / |mix|  (0 = perfect)
      tgt_energy_s{N}       — mean RMS² of each target channel
      est_energy_s{N}       — mean RMS² of each estimate channel
      mean_leakage          — avg |corr(est[c], tgt[other])| across active pairs
      mask_contrast         — std across channels (0 = all masks identical)
      mask_saturation       — fraction of mask values near 0 or 1
      mask_mean_s{N}        — average mask value per channel
      mask_std_s{N}         — std of mask values per channel
      spike_rate_layer{N}   — mean firing rate per SNN layer
      spike_rate_mean       — average across all layers
    """
    B, C, T = estimates.shape
    est = estimates.float()
    tgt = targets.float()
    mix = mixture.float()

    active = _active_channels(tgt)   # (C,) bool

    result: Dict[str, float] = {}

    # ── 1. Per-channel SI-SDRi (direct, no PIT) ──────────────────────────────
    for c in range(C):
        if active[c].item():
            sisdr_e = si_sdr(est[:, c], tgt[:, c])   # (B,)
            sisdr_m = si_sdr(mix,       tgt[:, c])   # (B,)
            result[f"si_sdri_s{c+1}"] = (sisdr_e - sisdr_m).mean().item()

    # ── 2. Sum-to-mixture reconstruction fidelity ─────────────────────────────
    recon    = est.sum(dim=1)
    recon_l1 = (recon - mix).abs().mean().item()
    mix_l1   = mix.abs().mean().item()
    result["reconstruction_l1_rel"] = recon_l1 / (mix_l1 + 1e-8)

    # ── 3. Per-channel energy ─────────────────────────────────────────────────
    tgt_rms = tgt.pow(2).mean(dim=-1)   # (B, C)
    est_rms = est.pow(2).mean(dim=-1)
    for c in range(C):
        result[f"tgt_energy_s{c+1}"] = tgt_rms[:, c].mean().item()
        result[f"est_energy_s{c+1}"] = est_rms[:, c].mean().item()

    # ── 4. Mean cross-channel leakage (active channels only) ─────────────────
    active_idx = [c for c in range(C) if active[c].item()]
    leakage_vals = []
    for c in active_idx:
        for other in active_idx:
            if other == c:
                continue
            e_c = est[:, c] - est[:, c].mean(-1, keepdim=True)
            t_o = tgt[:, other] - tgt[:, other].mean(-1, keepdim=True)
            corr = (e_c * t_o).sum(-1) / (
                e_c.norm(dim=-1) * t_o.norm(dim=-1) + 1e-8
            )
            leakage_vals.append(corr.mean().item())
    if leakage_vals:
        result["mean_leakage"] = float(np.mean(np.abs(leakage_vals)))

    # ── 5. Mask diagnostics ───────────────────────────────────────────────────
    if masks is not None:
        m = masks.float()   # (B, C, N, L) or (B, C, L)
        for c in range(C):
            mc = m[:, c]
            result[f"mask_mean_s{c+1}"] = mc.mean().item()
            result[f"mask_std_s{c+1}"]  = mc.std().item()
        # std across channels: 0 = all masks identical (model not separating)
        result["mask_contrast"]   = m.std(dim=1).mean().item()
        result["mask_saturation"] = ((m < 0.05) | (m > 0.95)).float().mean().item()

    # ── 6. Spike rate per layer ───────────────────────────────────────────────
    if spike_recs is not None and len(spike_recs) > 0:
        rates = []
        for i, rec in enumerate(spike_recs):
            rate = rec.float().mean().item()
            result[f"spike_rate_layer{i+1}"] = rate
            rates.append(rate)
        result["spike_rate_mean"] = float(np.mean(rates))

    return result


def print_diagnostics(d: Dict[str, float]) -> None:
    """Pretty-print a diagnostic_metrics dict."""
    sep = "─" * 62
    print(f"\n{sep}")
    print("  DIAGNOSTIC REPORT")
    print(sep)

    # Per-channel SI-SDRi (active channels only)
    si_keys = sorted(k for k in d if k.startswith("si_sdri_s"))
    if si_keys:
        print(f"\n  Per-channel SI-SDRi (improvement over mixture):")
        for k in si_keys:
            ch = k.split("si_sdri_s")[-1]
            bar = "▓" * max(0, int((d[k] + 20) / 2))
            print(f"    ch{ch}: {d[k]:+6.2f} dB   {bar}")

    # Reconstruction
    rel = d.get("reconstruction_l1_rel", float("nan"))
    verdict = ("good" if rel < 0.1 else "moderate" if rel < 0.3 else "poor")
    print(f"\n  Sum-to-mixture error (rel L1):  {rel:.3f}  ({verdict})")

    # Per-channel energy (all channels)
    print(f"\n  Channel energy (target vs estimate):")
    c = 1
    while f"tgt_energy_s{c}" in d:
        te = d[f"tgt_energy_s{c}"]
        ee = d.get(f"est_energy_s{c}", float("nan"))
        silent = "(silent)" if te < 1e-8 else ""
        print(f"    ch{c}: tgt={te:.4f}  est={ee:.4f}  {silent}")
        c += 1

    # Mean leakage
    if "mean_leakage" in d:
        lk = d["mean_leakage"]
        verdict = ("clean" if lk < 0.1 else "moderate" if lk < 0.3 else "high bleed")
        print(f"\n  Mean cross-channel leakage: {lk:.4f}  ({verdict})")

    # Masks
    if "mask_contrast" in d:
        mc = d["mask_contrast"]
        ms = d.get("mask_saturation", float("nan"))
        verdict = ("separating" if mc > 0.05 else
                   "weak"       if mc > 0.01 else "NOT SEPARATING")
        print(f"\n  Mask diagnostics:")
        print(f"    std across channels: {mc:.4f}  ({verdict})")
        print(f"    saturation (binary): {ms:.1%}")
        ci = 1
        while f"mask_mean_s{ci}" in d:
            print(f"    ch{ci}: mean={d[f'mask_mean_s{ci}']:.3f}"
                  f"  std={d.get(f'mask_std_s{ci}', float('nan')):.3f}")
            ci += 1

    # Spike rates
    if "spike_rate_mean" in d:
        print(f"\n  SNN spike rates (target ≈ 0.10):")
        i = 1
        while f"spike_rate_layer{i}" in d:
            rate = d[f"spike_rate_layer{i}"]
            verdict = ("dead" if rate < 0.01 else
                       "OK"   if rate < 0.25 else "saturated")
            print(f"    layer {i}: {rate:.4f}  ({verdict})")
            i += 1
        print(f"    mean:    {d['spike_rate_mean']:.4f}")

    print(f"{sep}\n")
