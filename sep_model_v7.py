"""
sep_model_v7.py — Hybrid SNN encoder + non-spiking convolutional decoder.

v7 change vs v6:
  The single ConvTranspose1d decoder is replaced by ConvDecoder — a
  multi-layer 1D convolutional block that refines masked encoder features
  before the final upsample.  This targets the 'hissing' artifact observed
  in v6 output, which is caused by high-frequency spike-pattern noise
  leaking through the single-layer transposed conv without suppression.

Architecture:
  Encoder       → Conv1d + ReLU          (unchanged from v1–v6)
  Separator     → ChunkedSNNSeparator    (LIF neurons, unchanged from v6)
  Decoder [NEW] → ConvDecoder            (n_refine × Conv1d + GroupNorm + PReLU,
                                          then ConvTranspose1d)

Spike → continuous transition (backprop note):
  LIFResidualBlock returns `spk + x` (residual), not raw binary spikes.
  ChunkedSNNSeparator concatenates this residual output with the continuous
  membrane potential, projects through out_proj, then applies softmax →
  masks ∈ (0, 1).  All values entering ConvDecoder are continuous; standard
  autograd applies throughout.  The only non-differentiable point is the LIF
  threshold step inside LIFResidualBlock, handled by snntorch's surrogate
  gradient (fast_sigmoid, slope=25).  Gradient flows:

    ConvDecoder ← masked enc features ← softmax masks ← out_proj
    ← membrane/spike readout ← surrogate_grad(LIF threshold)
    ← membrane dynamics ← in_proj ← encoder features

Weight transfer from v6:
  transfer_encoder_decoder() copies encoder weights as before.
  For the decoder: if src has a plain ConvTranspose1d, its weight is copied
  into dst.decoder.upsample (the equivalent layer in ConvDecoder).
  The refine blocks are freshly initialised — they are new capacity.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as _weight_norm

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNN = True
except ImportError:
    HAS_SNN = False


def _spike_grad():
    return surrogate.fast_sigmoid(slope=25) if HAS_SNN else None


def _safe_groups(n_filters: int, target: int = 8) -> int:
    """Largest divisor of n_filters that is <= target (minimum 1)."""
    for g in range(target, 0, -1):
        if n_filters % g == 0:
            return g
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# SpecAugment  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """Randomly zeros freq-axis bands of encoder output during training."""

    def __init__(self, freq_mask_param: int = 27, num_masks: int = 2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks       = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, N, L = x.shape
        mask = torch.ones(1, N, 1, dtype=x.dtype, device=x.device)
        for _ in range(self.num_masks):
            f  = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, N - f), (1,)).item()
            mask[:, f0:f0 + f, :] = 0.0
        return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# LIF residual block  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class LIFResidualBlock(nn.Module):
    """
    One LIF layer with residual connection.

    Returns (spk + x, spk, mem_next):
      - spk + x  : continuous residual output (what flows to the next layer)
      - spk      : binary spike for recordings / rate loss
      - mem_next : membrane state for the next time step

    The residual `spk + x` ensures gradients bypass the threshold even when
    the surrogate is weak — same motivation as ResNet skip connections.
    """

    def __init__(self, dim: int, beta: float = 0.9, threshold: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.fc   = nn.Linear(dim, dim, bias=False)
        self.bn   = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(p=dropout)
        self.dim  = dim

        if HAS_SNN:
            self.lif = snn.Leaky(
                beta=beta, threshold=threshold,
                spike_grad=_spike_grad(),
                learn_beta=True, learn_threshold=True,
            )
        else:
            self.register_buffer("_beta", torch.tensor(beta))
            self.threshold = threshold

    def init_mem(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.dim, device=device)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        if HAS_SNN and isinstance(getattr(self.lif, 'beta', None), torch.Tensor):
            self.lif.beta.data.clamp_(0.5, 0.99)
        cur = self.drop(self.bn(self.fc(x)))
        if HAS_SNN:
            spk, mem_next = self.lif(cur, mem)
        else:
            mem_next = self._beta * mem + cur
            spk      = (mem_next >= self.threshold).float()
            mem_next = mem_next * (1.0 - spk.detach())
        return spk + x, spk, mem_next


# ─────────────────────────────────────────────────────────────────────────────
# Shared segmentation helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def segment_sequence(x: torch.Tensor, chunk_size: int,
                     hop_size: int) -> Tuple[torch.Tensor, int, int, int]:
    B, N, L = x.shape
    seg, hop = chunk_size, hop_size
    n_segs   = max(1, (L - seg + hop - 1) // hop + 1)
    L_pad    = (n_segs - 1) * hop + seg
    if L_pad > L:
        x = F.pad(x, (0, L_pad - L))
    x_segs = x.unfold(-1, seg, hop)
    x_segs = x_segs.permute(0, 2, 3, 1)
    x_flat = x_segs.reshape(B * n_segs, seg, N)
    return x_flat, B, n_segs, L_pad


def reconstruct_sequence(h: torch.Tensor, B: int, n_segs: int,
                         chunk_size: int, hop_size: int,
                         L_pad: int, L_orig: int) -> torch.Tensor:
    seg, hop = chunk_size, hop_size
    out_dim  = h.shape[-1]
    device   = h.device

    h      = h.reshape(B, n_segs, seg, out_dim)
    output = torch.zeros(B, out_dim, L_pad, device=device)
    count  = torch.zeros(1, 1,       L_pad, device=device)

    for i in range(n_segs):
        start = i * hop
        output[:, :, start:start + seg] += h[:, i, :, :].permute(0, 2, 1)
        count[:,  :, start:start + seg] += 1.0

    return (output / count.clamp(min=1.0))[:, :, :L_orig]


# ─────────────────────────────────────────────────────────────────────────────
# ChunkedGRUSeparator  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedGRUSeparator(nn.Module):
    def __init__(self, n_filters: int, hidden: int, n_layers: int,
                 n_speakers: int = 2, dropout: float = 0.1,
                 chunk_size: int = 250):
        super().__init__()
        self.n_speakers = n_speakers
        self.n_filters  = n_filters
        self.chunk_size = chunk_size
        self.hop_size   = chunk_size // 2

        self.in_norm  = nn.LayerNorm(n_filters)
        self.gru      = nn.GRU(
            input_size    = n_filters,
            hidden_size   = hidden,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.out_proj = nn.Linear(hidden * 2, n_filters * n_speakers)

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape
        x_norm  = self.in_norm(enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_flat, B, n_segs, L_pad = segment_sequence(
            x_norm, self.chunk_size, self.hop_size)

        h, _  = self.gru(x_flat)
        out   = self.out_proj(h)

        out_full = reconstruct_sequence(
            out, B, n_segs, self.chunk_size, self.hop_size, L_pad, L)
        masks = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)
        return masks, []


# ─────────────────────────────────────────────────────────────────────────────
# ChunkedSNNSeparator  (unchanged — spiking latent layers)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedSNNSeparator(nn.Module):
    """
    LIF spiking separator — the spiking latent layers kept in v7.

    Spike→continuous transition: LIFResidualBlock outputs `spk + x` (residual),
    so all inter-block signals are continuous.  Surrogate gradient
    (fast_sigmoid) handles the binary threshold.  The final out_proj +
    softmax gives continuous masks ∈ (0,1) for the decoder.
    """

    def __init__(self, n_filters: int, hidden: int, n_layers: int,
                 n_speakers: int = 2, beta: float = 0.9,
                 threshold: float = 1.0, dropout: float = 0.1,
                 chunk_size: int = 100):
        super().__init__()
        self.n_speakers = n_speakers
        self.n_filters  = n_filters
        self.chunk_size = chunk_size
        self.hop_size   = chunk_size // 2

        self.in_proj = nn.Sequential(
            nn.LayerNorm(n_filters),
            nn.Linear(n_filters, hidden, bias=False),
        )
        self.blocks   = nn.ModuleList([
            LIFResidualBlock(hidden, beta, threshold, dropout)
            for _ in range(n_layers)
        ])
        self.mem_norm  = nn.LayerNorm(hidden)
        self.out_proj  = nn.Linear(hidden * 2, n_filters * n_speakers)

    def _process_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, List]:
        batch, seg, N = chunk.shape
        h_proj = self.in_proj(chunk)

        mems   = [blk.init_mem(batch, chunk.device) for blk in self.blocks]
        recs   = [[] for _ in self.blocks]
        h_outs = []
        m_outs = []

        for t in range(seg):
            h = h_proj[:, t, :]
            for i, blk in enumerate(self.blocks):
                h, spk, mems[i] = blk(h, mems[i])
                recs[i].append(spk)
            h_outs.append(h)
            m_outs.append(mems[-1])

        h_stack = torch.stack(h_outs, dim=1)
        m_norm  = self.mem_norm(torch.stack(m_outs, dim=1))
        readout = torch.cat([h_stack, m_norm], dim=-1)
        out     = self.out_proj(readout)

        recs = [torch.stack(r, dim=1) for r in recs]
        return out, recs

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape
        x_flat, B, n_segs, L_pad = segment_sequence(
            enc, self.chunk_size, self.hop_size)
        out_flat, spike_recs = self._process_chunk(x_flat)
        out_full = reconstruct_sequence(
            out_flat, B, n_segs, self.chunk_size, self.hop_size, L_pad, L)
        masks = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)
        flat_recs = [r.reshape(-1, r.shape[-1]) for r in spike_recs]
        return masks, flat_recs


# ─────────────────────────────────────────────────────────────────────────────
# ConvDecoder  [NEW in v7]
# ─────────────────────────────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """
    Non-spiking multi-layer 1D convolutional decoder.

    Replaces the single ConvTranspose1d used in v1–v6.  The additional
    Conv1d refinement stages suppress high-frequency noise in the masked
    encoder features before the final upsample, directly addressing the
    hissing artifact observed in v6.

    Inputs to this module are fully continuous (masked encoder features),
    so standard autograd applies — no surrogate gradient needed here.

    Architecture:
        for _ in range(n_refine):
            Conv1d(N, N, kernel=3, padding=1)  — channel mixing + smoothing
            GroupNorm(n_groups, N)              — normalise across feature dim
            PReLU(N)                            — learnable per-channel slope
        ConvTranspose1d(N, 1, kernel_sz, stride)  — waveform reconstruction

    n_groups defaults to min(8, n_filters) rounded to a divisor of n_filters.
    """

    def __init__(self, n_filters: int, kernel_sz: int, stride: int,
                 n_refine: int = 3, n_groups: int = 8,
                 use_weight_norm: bool = False):
        super().__init__()

        n_groups = _safe_groups(n_filters, n_groups)

        blocks: List[nn.Module] = []
        for _ in range(n_refine):
            conv = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
            if use_weight_norm:
                conv = _weight_norm(conv, name='weight', dim=0)
            blocks.extend([
                conv,
                nn.GroupNorm(n_groups, n_filters),
                nn.PReLU(n_filters),
            ])
        self.refine = nn.Sequential(*blocks)

        _up = nn.ConvTranspose1d(
            n_filters, 1, kernel_size=kernel_sz, stride=stride, bias=False
        )
        self.upsample = _weight_norm(_up, name='weight', dim=0) if use_weight_norm else _up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, N, L) → (B*C, 1, T)"""
        return self.upsample(self.refine(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full model  (v7: uses ConvDecoder)
# ─────────────────────────────────────────────────────────────────────────────

class SNNTasNet(nn.Module):
    """
    SNN-TasNet v7: hybrid spiking encoder/separator + non-spiking ConvDecoder.

    snn_mode='gru' → ChunkedGRUSeparator  (pre-training on WHAM!)
    snn_mode='snn' → ChunkedSNNSeparator  (LIF fine-tuning, true spiking)

    The encoder and separator are identical to v6.  The decoder is now
    ConvDecoder (n_refine=3 by default) instead of a single ConvTranspose1d.
    """

    def __init__(
        self,
        n_filters:       int   = 256,
        kernel_sz:       int   = 32,
        stride:          int   = 16,
        hidden:          int   = 192,
        n_layers:        int   = 2,
        n_speakers:      int   = 2,
        beta:            float = 0.9,
        dropout:         float = 0.3,
        snn_mode:        str   = "gru",
        chunk_size:      int   = 250,
        snn_chunk:       int   = 100,
        freq_mask:       int   = 27,
        n_freq_masks:    int   = 2,
        use_weight_norm: bool  = False,
        decoder_refine:  int   = 3,   # v7: refinement depth in ConvDecoder
        decoder_groups:  int   = 8,   # v7: GroupNorm groups in ConvDecoder
    ):
        super().__init__()
        self.n_speakers   = n_speakers
        self.stride       = stride
        self.kernel_sz    = kernel_sz
        self.n_filters    = n_filters
        self.snn_mode     = snn_mode
        self.use_weight_norm = use_weight_norm

        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_sz, stride=stride, bias=False),
            nn.ReLU(),
        )

        self.spec_aug = SpecAugment(freq_mask_param=freq_mask, num_masks=n_freq_masks)

        if snn_mode == "snn":
            if not HAS_SNN:
                print("[model] snntorch not installed — falling back to GRU mode")
                snn_mode = "gru"
            else:
                self.separator = ChunkedSNNSeparator(
                    n_filters, hidden, n_layers, n_speakers,
                    beta, 1.0, dropout, snn_chunk,
                )

        if snn_mode == "gru":
            self.separator = ChunkedGRUSeparator(
                n_filters, hidden, n_layers, n_speakers, dropout, chunk_size
            )

        self.decoder = ConvDecoder(
            n_filters    = n_filters,
            kernel_sz    = kernel_sz,
            stride       = stride,
            n_refine     = decoder_refine,
            n_groups     = decoder_groups,
            use_weight_norm = use_weight_norm,
        )

    def forward(self, mixture: torch.Tensor) -> Dict:
        B, T = mixture.shape
        enc  = self.encoder(mixture.unsqueeze(1))       # (B, N, L)
        enc  = self.spec_aug(enc)
        masks, spike_recs = self.separator(enc)          # (B, C, N, L)

        B_i, N_i, L_enc = enc.shape
        masked    = enc.unsqueeze(1) * masks             # (B, C, N, L)
        flat_in   = masked.reshape(B_i * self.n_speakers, N_i, L_enc)
        separated = self.decoder(flat_in).squeeze(1)     # (B*C, T')
        separated = separated.reshape(B_i, self.n_speakers, -1)

        T_out = separated.shape[-1]
        if T_out > T:
            separated = separated[..., :T]
        elif T_out < T:
            separated = F.pad(separated, (0, T - T_out))

        return {"separated": separated, "masks": masks, "spike_recs": spike_recs}


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer  (updated for ConvDecoder)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_encoder_decoder(src: SNNTasNet, dst: SNNTasNet,
                              verbose: bool = True) -> SNNTasNet:
    """
    Copy encoder (and optionally decoder.upsample) from src into dst.

    Encoder transfer: identical to v1–v6.

    Decoder transfer: dst always has ConvDecoder.
      - If src also has ConvDecoder (v7→v7): full load_state_dict.
      - If src has a plain ConvTranspose1d (v1–v6→v7): copy the conv weight
        into dst.decoder.upsample; the refine blocks are freshly initialised.
        This gives the new upsample layer the benefit of pre-trained
        reconstruction weights while refine blocks learn from scratch.
    """
    assert src.n_filters == dst.n_filters, \
        f"n_filters mismatch: src={src.n_filters} dst={dst.n_filters}"
    assert src.kernel_sz == dst.kernel_sz, \
        f"kernel_sz mismatch: src={src.kernel_sz} dst={dst.kernel_sz}"

    dst.encoder.load_state_dict(src.encoder.state_dict())

    src_dec = src.decoder
    dst_dec = dst.decoder

    if isinstance(src_dec, ConvDecoder):
        # v7 → v7: direct transfer
        try:
            dst_dec.load_state_dict(src_dec.state_dict())
            if verbose:
                print("[transfer] Decoder: full ConvDecoder → ConvDecoder copy")
        except RuntimeError as e:
            if verbose:
                print(f"[transfer] Decoder copy failed ({e}); refine blocks kept fresh")
    else:
        # v1–v6 → v7: src.decoder is ConvTranspose1d (possibly weight-normed)
        # Copy upsample weights; leave refine blocks randomly initialised.
        try:
            # Plain ConvTranspose1d
            dst_dec.upsample.load_state_dict(src_dec.state_dict())
            if verbose:
                print("[transfer] Decoder: ConvTranspose1d → upsample (refine: fresh)")
        except RuntimeError:
            # weight_norm was applied to src decoder — reconstruct weight
            try:
                src_w = src_dec.weight.data
            except AttributeError:
                # weight_norm stores weight_g / weight_v
                g = src_dec.weight_g.data
                v = src_dec.weight_v.data
                norms = v.view(v.shape[0], -1).norm(dim=1, keepdim=True).view(-1, 1, 1)
                src_w = g * v / norms.clamp(min=1e-8)

            dst_up = dst_dec.upsample
            if hasattr(dst_up, 'weight_v'):
                # dst upsample also weight-normed
                norms = src_w.view(src_w.shape[0], -1).norm(dim=1)
                dst_up.weight_v.data.copy_(src_w)
                dst_up.weight_g.data.copy_(norms.view(-1, 1, 1))
                if verbose:
                    print("[transfer] Decoder: weight_norm → weight_norm upsample (refine: fresh)")
            else:
                dst_up.weight.data.copy_(src_w)
                if verbose:
                    print("[transfer] Decoder: weight_norm src → plain upsample (refine: fresh)")

    if verbose:
        enc_params = count_parameters(src.encoder)
        dec_params = count_parameters(dst.decoder)
        print(f"[transfer] Encoder: {enc_params:,} params copied")
        print(f"[transfer] Decoder (ConvDecoder): {dec_params:,} params  "
              f"(refine: {count_parameters(dst.decoder.refine):,} fresh, "
              f"upsample: {count_parameters(dst.decoder.upsample):,} transferred)")
        print(f"[transfer] Separator: freshly initialised")

    return dst


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    B, T = 2, 48000
    mix  = torch.randn(B, T)

    print("=== v7 GRU mode (pre-training) ===")
    gru_model = SNNTasNet(snn_mode="gru")
    print(f"  encoder:  {count_parameters(gru_model.encoder):,} params")
    print(f"  decoder:  {count_parameters(gru_model.decoder):,} params  "
          f"(refine={count_parameters(gru_model.decoder.refine):,} + "
          f"upsample={count_parameters(gru_model.decoder.upsample):,})")
    print(f"  total:    {count_parameters(gru_model):,} params")
    gru_model.train()
    t0 = time.time()
    out = gru_model(mix)
    print(f"  forward:  {time.time()-t0:.2f}s  out={out['separated'].shape}")

    if HAS_SNN:
        print("\n=== v7 SNN mode (fine-tuning) ===")
        snn_model = SNNTasNet(snn_mode="snn")
        print(f"  total:    {count_parameters(snn_model):,} params")
        transfer_encoder_decoder(gru_model, snn_model)
        snn_model.train()
        t0 = time.time()
        out = snn_model(mix)
        print(f"  forward:  {time.time()-t0:.2f}s  spikes={len(out['spike_recs'])}")
        print(f"  output:   {out['separated'].shape}")
    else:
        print("\n[!] snntorch not installed. Run: pip install snntorch")
