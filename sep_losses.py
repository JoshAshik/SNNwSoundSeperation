"""
sep_losses.py — SI-SDR + multi-resolution spectral loss with PIT + mask distillation.

Loss components:
  1. SI-SDR with PIT: time-domain waveform alignment
  2. Multi-resolution spectral loss: L1 on log-STFT at 256/512/1024 FFT sizes
  3. Silence penalty: penalise energy in silent channels
  4. Spike rate regularisation: keep firing rate near 10% (SNN mode)
  5. Mask distillation loss: PIT-aware MSE between student and teacher masks
     (used in knowledge distillation from GRU teacher to SNN student)
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR
# ─────────────────────────────────────────────────────────────────────────────

def si_sdr(estimate: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB. Inputs: (B, T) or (T,). Returns: (B,) or scalar."""
    estimate = estimate - estimate.mean(-1, keepdim=True)
    target   = target   - target.mean(-1, keepdim=True)
    dot      = (estimate * target).sum(-1, keepdim=True)
    t_energy = (target * target).sum(-1, keepdim=True) + eps
    s_tgt    = (dot / t_energy) * target
    e_noise  = estimate - s_tgt
    ratio    = (s_tgt * s_tgt).sum(-1) / ((e_noise * e_noise).sum(-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def pit_si_sdr_loss(estimates: torch.Tensor,
                    targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PIT SI-SDR loss for (B, C, T) estimates and targets.
    Returns: (mean_loss, best_perm_indices per sample)
    """
    B, C, T = estimates.shape
    perms   = list(itertools.permutations(range(C)))

    perm_losses = []
    for perm in perms:
        loss_perm = sum(
            -si_sdr(estimates[:, p], targets[:, c])
            for c, p in enumerate(perm)
        ) / C
        perm_losses.append(loss_perm)

    all_losses           = torch.stack(perm_losses, 0)   # (n_perms, B)
    best_per_sample, idx = all_losses.min(0)             # (B,)
    return best_per_sample.mean(), idx


# ─────────────────────────────────────────────────────────────────────────────
# Multi-resolution spectral loss
# ─────────────────────────────────────────────────────────────────────────────

_HANN_WINDOW_CACHE: dict = {}   # (n_fft, device_str) → window tensor


def _log_stft_magnitude(wav: torch.Tensor, n_fft: int,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute log-magnitude STFT for a batch of waveforms.

    wav:    (B, T)
    returns (B, n_fft//2+1, frames)
    """
    hop = n_fft // 4
    key = (n_fft, str(wav.device))
    if key not in _HANN_WINDOW_CACHE:
        _HANN_WINDOW_CACHE[key] = torch.hann_window(n_fft, device=wav.device)
    window = _HANN_WINDOW_CACHE[key]
    spec   = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    return torch.log(spec.abs() + eps)   # (B, F, T_spec)


def mr_spectral_loss(estimates: torch.Tensor, targets: torch.Tensor,
                     fft_sizes: List[int] = (256, 512, 1024),
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Multi-resolution spectral loss.

    For each FFT size, computes L1 distance between log-magnitude STFTs
    of estimate and target. Averaged across resolutions.

    estimates: (B, C, T)
    targets:   (B, C, T)
    returns:   scalar
    """
    B, C, T = estimates.shape
    total   = torch.tensor(0.0, device=estimates.device)

    for n_fft in fft_sizes:
        for c in range(C):
            e_spec = _log_stft_magnitude(estimates[:, c], n_fft, eps)
            t_spec = _log_stft_magnitude(targets[:, c],   n_fft, eps)
            total  = total + F.l1_loss(e_spec, t_spec)

    return total / (len(fft_sizes) * C)


def pit_spectral_loss(estimates: torch.Tensor,
                      targets: torch.Tensor,
                      fft_sizes: List[int] = (256, 512, 1024)) -> torch.Tensor:
    """
    PIT-wrapped multi-resolution spectral loss.
    Tries all channel permutations, picks assignment with lowest spectral loss.
    Uses same permutation logic as SI-SDR PIT so both losses agree on assignment.
    """
    B, C, T = estimates.shape
    perms   = list(itertools.permutations(range(C)))

    perm_losses = []
    for perm in perms:
        # Build permuted estimate tensor
        est_perm = torch.stack([estimates[:, p] for p in perm], dim=1)
        perm_losses.append(mr_spectral_loss(est_perm, targets, fft_sizes))

    return min(perm_losses)


# ─────────────────────────────────────────────────────────────────────────────
# Direct (no-PIT) losses for label-conditioned fixed-slot separation
# ─────────────────────────────────────────────────────────────────────────────

def direct_si_sdr_loss(estimates: torch.Tensor,
                       targets:   torch.Tensor) -> torch.Tensor:
    """
    Per-channel SI-SDR loss with no permutation search.
    Used when output channels map to fixed label categories.

    Only computes SI-SDR for channels where at least one sample in the batch
    has an active target (RMS > 1e-4). Silent-target channels contribute
    no SI-SDR gradient — the silence penalty handles them instead.

    estimates: (B, C, T)
    targets:   (B, C, T)
    returns:   scalar loss
    """
    B, C, T = estimates.shape
    # active[c] = True if ANY sample in the batch has signal in channel c
    active  = (targets.abs().mean(dim=-1) >= 1e-4).any(dim=0)   # (C,)
    if not active.any():
        return torch.tensor(0.0, device=estimates.device)
    # Vectorised: gather active channels, compute SI-SDR on (B*n_active, T) in one call
    act_idx = active.nonzero(as_tuple=True)[0]          # (n_active,)
    n_act   = act_idx.shape[0]
    est_a   = estimates[:, act_idx].reshape(B * n_act, T)
    tgt_a   = targets[:,   act_idx].reshape(B * n_act, T)
    return -si_sdr(est_a, tgt_a).mean()


def direct_spectral_loss(estimates: torch.Tensor,
                         targets:   torch.Tensor,
                         fft_sizes: List[int] = (256, 512, 1024)) -> torch.Tensor:
    """
    Direct (no-PIT) multi-resolution spectral loss.
    Only accumulates loss over channels with active targets in the batch.

    estimates: (B, C, T)
    targets:   (B, C, T)
    returns:   scalar loss
    """
    B, C, T = estimates.shape
    active = (targets.abs().mean(dim=-1) >= 1e-4).any(dim=0)   # (C,)
    if not active.any():
        return torch.tensor(0.0, device=estimates.device)

    # Gather active channels once, reshape to (B*n_active, T) for batched STFT.
    # One STFT call per FFT size instead of n_active separate calls.
    act_idx = active.nonzero(as_tuple=True)[0]     # (n_active,)
    n_act   = act_idx.shape[0]
    est_a   = estimates[:, act_idx].reshape(B * n_act, T)
    tgt_a   = targets[:,   act_idx].reshape(B * n_act, T)

    total = torch.tensor(0.0, device=estimates.device)
    for n_fft in fft_sizes:
        e_spec = _log_stft_magnitude(est_a, n_fft)
        t_spec = _log_stft_magnitude(tgt_a, n_fft)
        total  = total + F.l1_loss(e_spec, t_spec)

    return total / len(fft_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class SeparationLoss(nn.Module):
    """
    Combined loss:
      total = si_sdr_loss + lambda_spec * spectral_loss
            + lambda_silence * silence_penalty
            + lambda_rate * spike_rate_regularisation

    use_pit=True  (default): PIT over all permutations — for blind separation
    use_pit=False:            Direct per-channel loss — for labeled fixed-slot separation
                              lambda_silence raised to 0.3 to handle many silent channels

    SI-SDR (time domain): measures waveform alignment
    MR-Spec (freq domain): measures spectral quality, penalises smearing
    Silence penalty: penalises energy in channels where target is silent
    Spike rate (SNN mode only): keeps neuron firing rates near target
    """

    def __init__(
        self,
        lambda_rate:    float = 0.03,    # v2: raised from 0.005 to combat layer-1 saturation
        target_rate:    float = 0.1,
        lambda_silence: float = 0.1,
        lambda_spec:    float = 0.5,
        lambda_recon:   float = 0.5,    # v2: halved from 1.0 — v1 over-corrected (6-10x under)
        fft_sizes:      Tuple = (256, 512, 1024),
        use_pit:        bool  = True,
    ):
        super().__init__()
        self.lambda_rate    = lambda_rate
        self.target_rate    = target_rate
        self.lambda_spec    = lambda_spec
        self.lambda_recon   = lambda_recon
        self.fft_sizes      = list(fft_sizes)
        self.use_pit        = use_pit
        # For labeled separation (no PIT), many channels are silent so raise penalty
        self.lambda_silence = lambda_silence if use_pit else max(lambda_silence, 0.3)

    def forward(
        self,
        estimates:  torch.Tensor,           # (B, C, T)
        targets:    torch.Tensor,           # (B, C, T)
        spike_recs: List[torch.Tensor],
        mixture:    "torch.Tensor | None" = None,   # (B, T)
    ) -> dict:

        # ── SI-SDR loss ───────────────────────────────────────────────────────
        if self.use_pit:
            sep_loss, perms = pit_si_sdr_loss(estimates, targets)
            spec_loss = pit_spectral_loss(estimates, targets, self.fft_sizes)
        else:
            sep_loss = direct_si_sdr_loss(estimates, targets)
            spec_loss = direct_spectral_loss(estimates, targets, self.fft_sizes)
            perms = torch.zeros(estimates.shape[0], dtype=torch.long)

        # ── Silence penalty ──────────────────────────────────────────────────
        # Penalise energy in channels where the target is silent
        silence_mask = (targets.abs().mean(dim=-1) < 1e-4)   # (B, C)
        if silence_mask.any():
            silent_energy = estimates[silence_mask].pow(2).mean()
        else:
            silent_energy = torch.tensor(0.0, device=estimates.device)

        # ── Reconstruction loss ───────────────────────────────────────────────
        # sum(estimates) should equal mixture — enforces complementary masks
        # Normalised by mixture scale so the term is dimensionless (~rel L1)
        recon_loss = torch.tensor(0.0, device=estimates.device)
        if self.lambda_recon > 0 and mixture is not None:
            recon      = estimates.sum(dim=1)                        # (B, T)
            mix_scale  = mixture.abs().mean().clamp(min=1e-8)
            recon_loss = F.l1_loss(recon, mixture) / mix_scale

        # ── Spike rate regularisation (SNN mode only) ────────────────────────
        rate_loss = torch.tensor(0.0, device=estimates.device)
        if self.lambda_rate > 0 and spike_recs:
            for rec in spike_recs:
                rate      = rec.float().mean()
                rate_loss = rate_loss + (rate - self.target_rate) ** 2
            rate_loss = rate_loss / len(spike_recs)

        total = (sep_loss
                 + self.lambda_spec    * spec_loss
                 + self.lambda_silence * silent_energy
                 + self.lambda_recon   * recon_loss
                 + self.lambda_rate    * rate_loss)

        return {
            "total":        total,
            "si_sdr_loss":  sep_loss,
            "spec_loss":    spec_loss,
            "silence_loss": silent_energy,
            "recon_loss":   recon_loss,
            "rate_loss":    rate_loss,
            "best_perms":   perms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mask distillation loss (GRU teacher → SNN student)
# ─────────────────────────────────────────────────────────────────────────────

def pit_mask_distillation_loss(
    student_masks: torch.Tensor,
    teacher_masks: torch.Tensor,
) -> torch.Tensor:
    """
    PIT-aware mask distillation loss.

    Computes MSE between student and teacher masks, trying both channel
    permutations and picking the best alignment. This allows the student
    to learn from the teacher's soft mask targets regardless of which
    channel assignment PIT chose for each.

    student_masks: (B, C, N, L)
    teacher_masks: (B, C, N, L) — detached from teacher graph
    returns: scalar MSE loss (best permutation)
    """
    B, C, N, L = student_masks.shape
    perms = list(itertools.permutations(range(C)))

    perm_losses = []
    for perm in perms:
        loss = sum(
            F.mse_loss(student_masks[:, p], teacher_masks[:, c])
            for c, p in enumerate(perm)
        ) / C
        perm_losses.append(loss)

    return min(perm_losses)