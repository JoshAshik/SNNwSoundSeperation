"""
encoding.py – Binaural cochleagram computation and spike encoding.

Pipeline:
  Waveform (C=2, T)
      → Gammatone filterbank (per channel)   → (C, F=64, n_frames)
      → Log-energy compression               → (C, F, n_frames)  ∈ [0, 1]
      → Rate-coded Poisson spike trains      → (T_snn, C*F)

References:
  - Patterson et al. (1992) gammatone filterbank
  - Mahowald & Douglas (1991) silicon retina  (spike encoding inspiration)
  - Zenke & Ganguli (2018) SuperSpike learning rule
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioConfig, CochleagramConfig, EncoderConfig, audio_cfg, cochlea_cfg, enc_cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Gammatone filterbank (pure-PyTorch, batch-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def _erb_bandwidth(fc: float) -> float:
    """Equivalent Rectangular Bandwidth for centre frequency `fc` (Hz)."""
    return 24.7 * (4.37 * fc / 1000 + 1.0)


def _make_gammatone_filters(
    n_filters:   int,
    fmin:        float,
    fmax:        float,
    sample_rate: int,
    filter_len:  int = 512,
) -> torch.Tensor:
    """
    Build a bank of N gammatone filters using the ERB scale.
    Returns a real-valued filter bank of shape (N, filter_len).
    """
    # ERB-spaced centre frequencies
    erb_min = 21.4 * math.log10(1 + fmin / 229.0)
    erb_max = 21.4 * math.log10(1 + fmax / 229.0)
    erbs    = torch.linspace(erb_min, erb_max, n_filters)
    fcs     = 229.0 * (10 ** (erbs / 21.4) - 1)   # (N,)

    t = torch.arange(filter_len, dtype=torch.float32) / sample_rate  # (L,)

    order = 4
    filters = []
    for fc_val in fcs.tolist():
        bw  = _erb_bandwidth(fc_val)
        A   = (2 * math.pi * bw) ** order / math.factorial(order - 1)
        env = A * t ** (order - 1) * torch.exp(-2 * math.pi * bw * t)
        carrier = torch.cos(2 * math.pi * fc_val * t)
        h = env * carrier
        # Normalise to unit energy
        h = h / (h.pow(2).sum().sqrt() + 1e-8)
        filters.append(h)

    return torch.stack(filters, dim=0)   # (N, L)


class GammatoneFilterbank(nn.Module):
    """
    Differentiable gammatone filterbank implemented as 1-D convolution.

    Input:  (B, C, T) – batch of binaural waveforms
    Output: (B, C, N, n_frames) – per-channel, per-filter, per-frame energy
    """

    def __init__(
        self,
        cochlea_cfg: CochleagramConfig = cochlea_cfg,
        audio_cfg:   AudioConfig       = audio_cfg,
    ):
        super().__init__()
        self.n_filters   = cochlea_cfg.n_filters
        self.hop_samples = int(audio_cfg.sample_rate * cochlea_cfg.hop_ms / 1000)
        self.frame_samples = int(audio_cfg.sample_rate * cochlea_cfg.frame_ms / 1000)

        filters = _make_gammatone_filters(
            n_filters=cochlea_cfg.n_filters,
            fmin=cochlea_cfg.fmin,
            fmax=cochlea_cfg.fmax,
            sample_rate=audio_cfg.sample_rate,
            filter_len=self.frame_samples,
        )                                          # (N, L)
        # Store as conv weight: (out_channels=N, in_channels=1, kernel_size=L)
        self.register_buffer("weight", filters.unsqueeze(1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, C, T)
        Returns:
            cochleagram: (B, C, N, n_frames)  values ∈ [0, 1]
        """
        B, C, T = waveform.shape
        # Flatten channels → batch dimension for grouped conv
        x = waveform.view(B * C, 1, T)            # (B*C, 1, T)
        pad = self.frame_samples // 2
        x = F.pad(x, (pad, pad))

        # Convolve with filterbank
        out = F.conv1d(
            x,
            self.weight,                           # (N, 1, L)
            stride=self.hop_samples,
        )                                          # (B*C, N, n_frames)

        # Log-energy compression with global normalisation
        # IMPORTANT: do NOT normalise per-sample -- that destroys amplitude info
        # and makes silent (all-zero) inputs look identical to loud inputs.
        energy = out.pow(2)
        log_e  = torch.log1p(energy * 1e4)        # (B*C, N, n_frames)

        # Global sigmoid normalisation: maps log-energy to [0,1]
        # using a fixed scale so relative loudness is preserved across samples
        log_e = torch.sigmoid(log_e - log_e.mean())   # centre then sigmoid

        # Reshape → (B, C, N, n_frames)
        n_frames = log_e.shape[-1]
        return log_e.view(B, C, self.n_filters, n_frames)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Interaural cue extraction  (ILD + IPD)
# ──────────────────────────────────────────────────────────────────────────────

class InterauralFeatures(nn.Module):
    """
    Compute per-frequency-band Interaural Level Difference (ILD) and
    Interaural Phase Difference (IPD) features.

    These binaural cues help localise and separate sources spatially.

    Input:  (B, C=2, N, n_frames)
    Output: (B, 2*N, n_frames)  – ILD and IPD concatenated along feature axis
    """

    def forward(self, cochleagram: torch.Tensor) -> torch.Tensor:
        left  = cochleagram[:, 0]   # (B, N, F)
        right = cochleagram[:, 1]

        # ILD: level difference in dB (clamped for stability)
        ild = 20 * torch.log10((left + 1e-8) / (right + 1e-8)).clamp(-30, 30) / 30.0

        # IPD proxy: sign of cross-correlation (energy ratio)
        ipd = (left - right) / (left + right + 1e-8)

        return torch.cat([ild, ipd], dim=1)         # (B, 2*N, n_frames)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Spike encoder  (rate-coded Poisson)
# ──────────────────────────────────────────────────────────────────────────────

class PoissonSpikeEncoder(nn.Module):
    """
    Convert a normalised feature map into binary Poisson spike trains.

    The input features ∈ [0, 1] are treated as firing probabilities per
    SNN timestep.  The temporal dimension of the feature map is first
    average-pooled to exactly `T` SNN timesteps.

    Input:  (B, D, n_frames)   – D-dimensional feature per time frame
    Output: (T, B, D)          – binary spike tensor (0/1)
    """

    def __init__(self, enc_cfg: EncoderConfig = enc_cfg):
        super().__init__()
        self.T         = enc_cfg.T
        self.max_rate  = enc_cfg.max_rate
        self.threshold = enc_cfg.threshold

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, n_frames)  float ∈ [0, 1]
        Returns:
            rates:    (T, B, D)         deterministic firing rates (not binary spikes)

        NOTE: Using deterministic rate encoding instead of stochastic Poisson.
        With only ~300 training samples, stochastic encoding adds too much noise
        for the model to learn consistent feature-to-label mappings.
        The SNN LIF layers still process these as continuous inputs and produce
        binary spikes internally via the threshold mechanism.
        """
        B, D, n_frames = features.shape

        # Temporal down-sampling to T SNN steps
        if n_frames != self.T:
            rate = F.adaptive_avg_pool1d(features, self.T)   # (B, D, T)
        else:
            rate = features

        # Zero out sub-threshold activity, keep rest as continuous rates
        rate = rate.clamp(min=self.threshold).permute(2, 0, 1)  # (T, B, D)
        return rate


class TemporalContrastEncoder(nn.Module):
    """
    Temporal contrast (ON/OFF) spike encoding – mimics retinal ganglion cells.
    Emits ON spike when feature increases, OFF spike when it decreases.

    Input:  (B, D, n_frames)
    Output: (T, B, 2*D)   – ON and OFF channels interleaved
    """

    def __init__(self, enc_cfg: EncoderConfig = enc_cfg, threshold: float = 0.05):
        super().__init__()
        self.T         = enc_cfg.T
        self.threshold = threshold

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, D, n_frames = features.shape
        if n_frames != self.T:
            feat = F.adaptive_avg_pool1d(features, self.T)
        else:
            feat = features                                    # (B, D, T)

        diff = F.pad(feat.diff(dim=-1), (1, 0))               # (B, D, T)
        on_spikes  = (diff >  self.threshold).float()
        off_spikes = (diff < -self.threshold).float()

        # (T, B, 2D)
        out = torch.cat([on_spikes, off_spikes], dim=1).permute(2, 0, 1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 4.  End-to-end feature pipeline
# ──────────────────────────────────────────────────────────────────────────────

class AudioToSpikes(nn.Module):
    """
    Full pipeline: binaural waveform → spike trains.

    Steps:
      1. Gammatone filterbank  → cochleagram (B, C=2, N=64, n_frames)
      2. Interaural features   → ILD + IPD  (B, 2*N=128, n_frames)
      3. Monaural cochleagram  → flattened  (B, 2*N=128, n_frames)  [merged]
      4. Concat               →             (B, 4*N=256, n_frames)  — wait, see below
      5. Spike encode          →             (T, B, D)

    Final spike input dimension D = C*N + 2*N = 2*64 + 2*64 = 256
    (monaural cochleagram + ILD + IPD)
    """

    def __init__(
        self,
        cochlea_cfg: CochleagramConfig = cochlea_cfg,
        audio_cfg:   AudioConfig       = audio_cfg,
        enc_cfg:     EncoderConfig     = enc_cfg,
        encoder_type: str              = "poisson",   # "poisson" | "contrast"
    ):
        super().__init__()
        self.filterbank  = GammatoneFilterbank(cochlea_cfg, audio_cfg)
        self.iau_feats   = InterauralFeatures()

        if encoder_type == "contrast":
            self.spike_enc = TemporalContrastEncoder(enc_cfg)
            self.out_dim   = (2 * cochlea_cfg.n_filters) + (2 * 2 * cochlea_cfg.n_filters)
        else:
            self.spike_enc = PoissonSpikeEncoder(enc_cfg)
            self.out_dim   = cochlea_cfg.n_filters * 2 + 2 * cochlea_cfg.n_filters   # 256

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, C=2, T)
        Returns:
            spikes:   (T_snn, B, D=256)
        """
        cochleagram = self.filterbank(waveform)                     # (B, 2, 64, F)

        # Flatten monaural energy: L + R channels → (B, 128, n_frames)
        B, C, N, n_frames = cochleagram.shape
        mono = cochleagram.view(B, C * N, n_frames)                 # (B, 128, n_frames)

        # Binaural cues
        iau = self.iau_feats(cochleagram)                           # (B, 128, n_frames)

        # Concatenate all features → (B, 256, n_frames)
        combined = torch.cat([mono, iau], dim=1)

        # Normalise to [0, 1]
        combined = combined.clamp(0, 1)

        # Encode to spikes → (T, B, 256)
        return self.spike_enc(combined)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe = AudioToSpikes()
    dummy = torch.randn(4, 2, 48_000)   # (B=4, C=2, T=48000)
    spikes = pipe(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Spikes: {spikes.shape}")   # expected (100, 4, 256)
    print(f"Spike rate: {spikes.mean():.3f}")