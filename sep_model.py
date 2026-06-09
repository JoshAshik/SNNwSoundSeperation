"""
sep_model.py — SNN-TasNet with chunked SNN separator and weight transfer.

Two separator backends:
  snn_mode='gru' → ChunkedGRUSeparator  (fast, used for WHAM! pre-training)
  snn_mode='snn' → ChunkedSNNSeparator  (true LIF, used for fine-tuning)

Key design: the encoder and decoder are IDENTICAL between GRU and SNN modes.
This means pre-trained encoder/decoder weights transfer perfectly — only the
separator changes when switching from GRU pre-training to SNN fine-tuning.

ChunkedSNNSeparator:
  Same chunk/overlap-add scheme as ChunkedGRUSeparator, but each chunk is
  processed by LIF neurons frame-by-frame instead of a GRU. Because chunks
  are independent they run in parallel as a large batch, giving ~chunk_size×
  speedup over full-sequence frame-by-frame processing. LIF membrane dynamics
  reset to zero at each chunk boundary, which is biologically plausible
  (no long-range membrane state carries across the overlap boundary).

Weight transfer function:
  transfer_encoder_decoder(src_model, dst_model) copies encoder and decoder
  weights from a GRU-mode pre-trained model into an SNN-mode model, leaving
  only the separator weights freshly initialised.
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


# ─────────────────────────────────────────────────────────────────────────────
# SpecAugment
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
        # Build a (1, N, 1) freq mask — much smaller than cloning (B, N, L)
        mask = torch.ones(1, N, 1, dtype=x.dtype, device=x.device)
        for _ in range(self.num_masks):
            f  = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, N - f), (1,)).item()
            mask[:, f0:f0 + f, :] = 0.0
        return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# LIF residual block
# ─────────────────────────────────────────────────────────────────────────────

class LIFResidualBlock(nn.Module):
    """
    One layer of LIF neurons with residual connection.

    Input/output: (B, dim)
    Membrane: (B, dim) — passed in and out so caller controls state across time.

    BatchNorm1d operates on (B, dim) — treats each neuron as a channel.
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
        # Clamp beta so membrane decay stays bounded. v1 drifted to beta=1.0001
        # (pure integrator, no decay) causing layer-1 spike rate saturation.
        if HAS_SNN and isinstance(getattr(self.lif, 'beta', None), torch.Tensor):
            self.lif.beta.data.clamp_(0.5, 0.99)
        cur = self.drop(self.bn(self.fc(x)))
        if HAS_SNN:
            spk, mem_next = self.lif(cur, mem)
        else:
            mem_next = self._beta * mem + cur
            spk      = (mem_next >= self.threshold).float()
            mem_next = mem_next * (1.0 - spk.detach())   # hard reset
        return spk + x, spk, mem_next   # residual output, true spike (binary), new mem


# ─────────────────────────────────────────────────────────────────────────────
# Shared segmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def segment_sequence(x: torch.Tensor, chunk_size: int,
                     hop_size: int) -> Tuple[torch.Tensor, int, int, int]:
    """
    Segment (B, N, L) into overlapping chunks.
    Returns (x_flat, B, n_segs, L_pad) where x_flat is (B*n_segs, seg, N).
    """
    B, N, L = x.shape
    seg, hop = chunk_size, hop_size
    n_segs   = max(1, (L - seg + hop - 1) // hop + 1)
    L_pad    = (n_segs - 1) * hop + seg
    if L_pad > L:
        x = F.pad(x, (0, L_pad - L))
    x_segs = x.unfold(-1, seg, hop)                  # (B, N, n_segs, seg)
    x_segs = x_segs.permute(0, 2, 3, 1)              # (B, n_segs, seg, N)
    x_flat = x_segs.reshape(B * n_segs, seg, N)      # (B*n_segs, seg, N)
    return x_flat, B, n_segs, L_pad


def reconstruct_sequence(h: torch.Tensor, B: int, n_segs: int,
                         chunk_size: int, hop_size: int,
                         L_pad: int, L_orig: int) -> torch.Tensor:
    """
    Overlap-add (B*n_segs, seg, out_dim) → (B, out_dim, L_orig).
    """
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
# Chunked GRU separator (pre-training backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedGRUSeparator(nn.Module):
    """
    Bidirectional GRU applied independently to overlapping chunks.
    Fast, vectorised. Used for WHAM! pre-training.
    """

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
# Chunked SNN separator (fine-tuning backbone — true spiking neurons)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedSNNSeparator(nn.Module):
    """
    LIF spiking neurons applied chunk-by-chunk.

    Why chunked instead of full-sequence:
      Processing all L≈3000 frames as a single sequential LIF loop is
      impractical (251s/epoch as observed). Splitting into chunks of
      `chunk_size` frames lets the B*n_segs chunks run as a large
      parallelised batch. Within each chunk, LIF dynamics are genuine
      sequential spiking with membrane state. The membrane resets to zero
      at chunk boundaries — a biologically plausible "refractory period"
      between segments.

    Speedup vs full-sequence SNN:
      Full sequence: L=3000 serial steps, batch=B
      Chunked:       chunk_size=100 serial steps, batch=B*n_segs
      Speedup ≈ L/chunk_size = 30× (from ~250s/epoch to ~8s/epoch)

    Architecture per chunk:
      in_proj: (N,) → (hidden,)   [LayerNorm + Linear]
      n_layers of LIFResidualBlock
      membrane readout: concat(spike_output, LayerNorm(membrane)) → (hidden*2,)
      out_proj: (hidden*2,) → (N*C,)

    Output: masks (B, C, N, L) + spike recordings List[Tensor]
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
        # Membrane readout: output projection sees BOTH spike output (binary+residual)
        # and membrane state (continuous) for richer mask generation.
        # The membrane potential carries temporal dynamics that binary spikes alone
        # cannot capture — critical for producing good continuous-valued masks.
        self.mem_norm  = nn.LayerNorm(hidden)
        self.out_proj  = nn.Linear(hidden * 2, n_filters * n_speakers)

    def _process_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Process one batch of chunks through LIF neurons.

        chunk: (batch, seg, N) where batch = B*n_segs
        Returns: (batch, seg, N*C), spike_recordings

        Speedup vs original: in_proj and out_proj are moved outside the time loop.
        PyTorch Linear/LayerNorm support arbitrary leading dims, so applying them
        to (batch, seg, dim) is equivalent to calling them seg times on (batch, dim)
        but executes as one large GPU matmul instead of seg small sequential ones.
        The only part that must remain sequential is the LIF membrane update, which
        has strict time-step dependencies.
        """
        batch, seg, N = chunk.shape

        # Apply in_proj to all time steps at once — no time dependencies
        h_proj = self.in_proj(chunk)          # (batch, seg, hidden)

        mems   = [blk.init_mem(batch, chunk.device) for blk in self.blocks]
        recs   = [[] for _ in self.blocks]
        h_outs = []
        m_outs = []

        for t in range(seg):
            h = h_proj[:, t, :]               # (batch, hidden)
            for i, blk in enumerate(self.blocks):
                h, spk, mems[i] = blk(h, mems[i])
                recs[i].append(spk)
            h_outs.append(h)
            m_outs.append(mems[-1])

        # Apply mem_norm and out_proj to all time steps in one matmul each
        h_stack = torch.stack(h_outs, dim=1)                      # (batch, seg, hidden)
        m_norm  = self.mem_norm(torch.stack(m_outs, dim=1))       # (batch, seg, hidden)
        readout = torch.cat([h_stack, m_norm], dim=-1)            # (batch, seg, hidden*2)
        out     = self.out_proj(readout)                          # (batch, seg, N*C)

        recs = [torch.stack(r, dim=1) for r in recs]
        return out, recs

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape

        # Segment encoder output into overlapping chunks
        # x_flat: (B*n_segs, chunk_size, N)
        x_flat, B, n_segs, L_pad = segment_sequence(
            enc, self.chunk_size, self.hop_size)

        # Run LIF through each frame within each chunk (all chunks batched)
        out_flat, spike_recs = self._process_chunk(x_flat)
        # out_flat: (B*n_segs, chunk_size, N*C)

        # Overlap-add reconstruction → (B, N*C, L)
        out_full = reconstruct_sequence(
            out_flat, B, n_segs, self.chunk_size, self.hop_size, L_pad, L)

        masks = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)

        # Aggregate spike recordings: average across chunks for monitoring
        # spike_recs[layer]: (B*n_segs, chunk_size, hidden) → flatten for loss
        flat_recs = [r.reshape(-1, r.shape[-1]) for r in spike_recs]
        return masks, flat_recs


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class SNNTasNet(nn.Module):
    """
    SNN-TasNet with switchable separator.

    snn_mode='gru' → ChunkedGRUSeparator  (pre-training on WHAM!)
    snn_mode='snn' → ChunkedSNNSeparator  (fine-tuning, true spiking)

    Encoder and decoder are identical in both modes — weights transfer cleanly.
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
        chunk_size:      int   = 250,   # GRU chunks (large = more context)
        snn_chunk:       int   = 100,   # SNN chunks (smaller = manageable LIF loop)
        freq_mask:       int   = 27,
        n_freq_masks:    int   = 2,
        use_weight_norm: bool  = False, # v5+: weight_norm on decoder (training stabiliser)
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.stride     = stride
        self.kernel_sz  = kernel_sz
        self.n_filters  = n_filters
        self.snn_mode   = snn_mode

        # Encoder: same in both modes
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_sz, stride=stride, bias=False),
            nn.ReLU(),
        )

        self.spec_aug = SpecAugment(freq_mask_param=freq_mask, num_masks=n_freq_masks)

        if snn_mode == "snn":
            if not HAS_SNN:
                print("[model] snntorch not installed — install with: pip install snntorch")
                print("[model] Falling back to GRU mode")
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

        # Decoder: same in both modes.
        # use_weight_norm=True (v5+): decomposes weight = g * v/||v||.
        # Separating magnitude (g) from direction (v) stabilises optimisation
        # and prevents the decoder from silently growing large norms that
        # bypass the reconstruction loss.  g is still learned — the MSE
        # reconstruction loss is what actually constrains the amplitude.
        _dec = nn.ConvTranspose1d(
            n_filters, 1, kernel_size=kernel_sz, stride=stride, bias=False
        )
        self.decoder = _weight_norm(_dec, name="weight", dim=0) if use_weight_norm else _dec
        self.use_weight_norm = use_weight_norm

    def forward(self, mixture: torch.Tensor) -> Dict:
        B, T = mixture.shape
        enc  = self.encoder(mixture.unsqueeze(1))   # (B, N, L)
        enc  = self.spec_aug(enc)
        masks, spike_recs = self.separator(enc)     # (B, C, N, L)

        # Vectorised decode: one ConvTranspose1d call at batch B*C instead of C calls at B.
        # enc.unsqueeze(1) broadcasts against masks (B, C, N, L) to give (B, C, N, L).
        B_i, N_i, L_enc = enc.shape
        masked     = enc.unsqueeze(1) * masks                              # (B, C, N, L)
        separated  = self.decoder(
            masked.reshape(B_i * self.n_speakers, N_i, L_enc)
        ).squeeze(1).reshape(B_i, self.n_speakers, -1)                     # (B, C, T')
        T_out = separated.shape[-1]
        if T_out > T:
            separated = separated[..., :T]
        elif T_out < T:
            separated = F.pad(separated, (0, T - T_out))

        return {"separated": separated, "masks": masks, "spike_recs": spike_recs}


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer
# ─────────────────────────────────────────────────────────────────────────────

def transfer_encoder_decoder(src: SNNTasNet, dst: SNNTasNet,
                              verbose: bool = True) -> SNNTasNet:
    """
    Copy encoder and decoder weights from src (GRU pre-trained) into dst (SNN).

    The separator is NOT copied — it is left freshly initialised in dst.
    This is the core of transfer learning: the encoder has learned to extract
    meaningful audio features from WHAM!, and those features transfer directly
    to the 300-scene separation task.

    src and dst must have matching n_filters, kernel_sz, stride.
    """
    assert src.n_filters == dst.n_filters, \
        f"n_filters mismatch: src={src.n_filters} dst={dst.n_filters}"
    assert src.kernel_sz == dst.kernel_sz, \
        f"kernel_sz mismatch: src={src.kernel_sz} dst={dst.kernel_sz}"

    # Copy encoder
    dst.encoder.load_state_dict(src.encoder.state_dict())

    # Copy decoder — handle weight_norm if applied to dst.decoder.
    # weight_norm replaces 'weight' with 'weight_g' + 'weight_v'; direct
    # load_state_dict from a plain checkpoint would fail with key mismatch.
    try:
        dst.decoder.load_state_dict(src.decoder.state_dict())
    except RuntimeError:
        # dst.decoder has weight_norm applied — initialise g/v from src weight.
        # After init: effective weight = g * v/||v|| = norms * w/norms = w  ✓
        src_w = src.decoder.weight.data          # (in_ch, out_ch, kernel)
        norms = src_w.view(src_w.shape[0], -1).norm(dim=1)   # (in_ch,)
        dst.decoder.weight_v.data.copy_(src_w)
        dst.decoder.weight_g.data.copy_(norms.view(-1, 1, 1))
        if verbose:
            print("[transfer] Decoder weight_norm detected — transferred via weight_g/v")

    if verbose:
        src_params = count_parameters(src)
        dst_params = count_parameters(dst)
        enc_params = count_parameters(src.encoder) + count_parameters(src.decoder)
        print(f"[transfer] Copied encoder+decoder: {enc_params:,} params")
        print(f"[transfer] Separator freshly initialised")
        print(f"[transfer] src total: {src_params:,}  dst total: {dst_params:,}")

    return dst


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time

    B, T = 2, 48000
    mix  = torch.randn(B, T)

    print("=== GRU mode (pre-training) ===")
    gru_model = SNNTasNet(snn_mode="gru")
    print(f"  params={count_parameters(gru_model):,}")
    gru_model.train()
    t0 = time.time()
    out = gru_model(mix)
    print(f"  forward: {time.time()-t0:.2f}s  out={out['separated'].shape}")

    if HAS_SNN:
        print("\n=== SNN mode (fine-tuning) ===")
        snn_model = SNNTasNet(snn_mode="snn")
        print(f"  params={count_parameters(snn_model):,}")
        transfer_encoder_decoder(gru_model, snn_model)
        snn_model.train()
        t0 = time.time()
        out = snn_model(mix)
        print(f"  forward: {time.time()-t0:.2f}s  spikes={len(out['spike_recs'])}")
    else:
        print("\n[!] snntorch not installed. Run: pip install snntorch")