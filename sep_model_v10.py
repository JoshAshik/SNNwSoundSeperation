"""
sep_model_v10.py — SNN-TasNet with stateful LIF separator.

v10 changes vs v7:
  Separator [REPLACED] → StatefulSNNSeparator
    - Chunks processed SEQUENTIALLY instead of batched in parallel
    - LIF membrane state carries across chunk boundaries (full 4s speaker memory)
    - State detached at chunk boundaries (TBPTT) — memory bounded to one chunk
    - Non-overlapping chunks (hop = chunk_size) for clean state handoff
    - In v7 the LIF neurons reset every 0.1 s; in v10 they see the full clip

  Encoder and ConvDecoder are unchanged from v7.

Why stateful LIF matters:
  Speech separation requires tracking which speaker is which across the full
  utterance. In v7/v8/v9, each 0.1 s chunk starts with zero membrane state —
  the model re-identifies speakers from scratch in every window. The persistent
  state here means a neuron that started firing for Speaker A in chunk 1 stays
  primed for Speaker A in chunks 2, 3, ... 40, giving the separator a coherent
  speaker timeline.

Architecture:
  Encoder    Conv1d(1→256, k=32, s=16, ReLU)           [unchanged from v7]
  Separator  StatefulSNNSeparator                        [NEW in v10]
               in_proj:   LayerNorm(256) → Linear(256→hidden)
               n_layers × LIFResidualBlock(hidden)
               mem_norm:  LayerNorm(hidden)
               out_proj:  Linear(hidden*2 → 256*2)
               sequential chunks, state carry, TBPTT detach
  Decoder    ConvDecoder(3× Conv1d+GN+PReLU → ConvTranspose1d)  [unchanged]

Weight transfer from v7:
  Encoder and decoder shapes are identical → copy directly.
  Separator dims change (hidden 256→512, n_layers 3→6) → freshly initialised.
  Use transfer_encoder_decoder(v7_model, v10_model) or the partial-key copy
  in two_speaker_train_v10.py.
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
    for g in range(target, 0, -1):
        if n_filters % g == 0:
            return g
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# SpecAugment  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
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
# LIF residual block  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

class LIFResidualBlock(nn.Module):
    """One LIF layer with residual connection."""

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
# Sequence segmentation helpers  (unchanged from v7)
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
# ChunkedGRUSeparator  (unchanged from v7 — used for GRU pre-train mode)
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
# ChunkedSNNSeparator  (unchanged from v7 — kept for reference / backwards compat)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedSNNSeparator(nn.Module):
    """Stateless LIF separator (v7 architecture). Chunks batched in parallel."""

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
        recs    = [torch.stack(r, dim=1) for r in recs]
        return out, recs

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape
        x_flat, B, n_segs, L_pad = segment_sequence(
            enc, self.chunk_size, self.hop_size)
        out_flat, spike_recs = self._process_chunk(x_flat)
        out_full = reconstruct_sequence(
            out_flat, B, n_segs, self.chunk_size, self.hop_size, L_pad, L)
        masks    = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)
        flat_recs = [r.reshape(-1, r.shape[-1]) for r in spike_recs]
        return masks, flat_recs


# ─────────────────────────────────────────────────────────────────────────────
# StatefulSNNSeparator  [NEW in v10]
# ─────────────────────────────────────────────────────────────────────────────

class StatefulSNNSeparator(nn.Module):
    """
    LIF spiking separator with cross-chunk membrane state persistence.

    Processing model:
      - Encoder output is split into non-overlapping chunks (hop = chunk_size).
        Non-overlapping gives each frame exactly one state context; overlapping
        would mean the same frame processed at two different membrane states.
      - Chunks are processed SEQUENTIALLY (outer loop over n_segs).
        This is the key difference from ChunkedSNNSeparator, which batches all
        chunks in parallel and resets state at each chunk boundary.
      - The membrane state from the end of chunk k is passed (detached) as the
        initial state for chunk k+1. Detach keeps gradient memory bounded to one
        chunk (TBPTT). The FORWARD PASS retains full temporal continuity —
        neurons maintain a coherent firing pattern for each speaker across the
        full 4 s clip.

    Performance note:
      Sequential processing reduces GPU utilisation per step (batch=B instead of
      B*n_segs). Epochs are expected to be 3–5× slower than v7. With 40 non-
      overlapping chunks per 4 s clip, the outer loop runs 40 iterations vs 80
      for the overlapping v7 design.
    """

    def __init__(self, n_filters: int, hidden: int, n_layers: int,
                 n_speakers: int = 2, beta: float = 0.9,
                 threshold: float = 1.0, dropout: float = 0.1,
                 chunk_size: int = 100):
        super().__init__()
        self.n_speakers = n_speakers
        self.n_filters  = n_filters
        self.chunk_size = chunk_size
        self.hop_size   = chunk_size     # non-overlapping: each frame processed once

        self.in_proj = nn.Sequential(
            nn.LayerNorm(n_filters),
            nn.Linear(n_filters, hidden, bias=False),
        )
        self.blocks = nn.ModuleList([
            LIFResidualBlock(hidden, beta, threshold, dropout)
            for _ in range(n_layers)
        ])
        self.mem_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden * 2, n_filters * n_speakers)

    def _process_chunk(
        self,
        chunk: torch.Tensor,             # (B, seg, N_filters)
        mems:  List[torch.Tensor],       # list[n_layers] of (B, hidden) — incoming state
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run one chunk through all LIF layers with provided initial state.

        Returns:
          out:        (B, seg, n_filters * n_speakers)
          spike_recs: list[n_layers] of (B, seg, hidden)
          mems:       list[n_layers] of (B, hidden) — state at end of chunk
        """
        B, seg, _ = chunk.shape
        h_proj = self.in_proj(chunk)     # (B, seg, hidden)

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

        h_stack = torch.stack(h_outs, dim=1)                # (B, seg, hidden)
        m_norm  = self.mem_norm(torch.stack(m_outs, dim=1)) # (B, seg, hidden)
        readout = torch.cat([h_stack, m_norm], dim=-1)
        out     = self.out_proj(readout)                     # (B, seg, out_dim)

        spike_recs = [torch.stack(r, dim=1) for r in recs]  # list of (B, seg, hidden)
        return out, spike_recs, mems

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape
        seg     = self.chunk_size
        hop     = self.hop_size       # == seg (non-overlapping)

        # Segment into non-overlapping chunks
        x_flat, B2, n_segs, L_pad = segment_sequence(enc, seg, hop)
        x_chunks = x_flat.reshape(B, n_segs, seg, N)

        # Zero-initialise membrane states — fresh at start of each utterance
        mems = [blk.init_mem(B, enc.device) for blk in self.blocks]

        all_outs: List[torch.Tensor]       = []
        all_spks: List[List[torch.Tensor]] = []   # [layer][chunk] → (B, seg, hidden)

        for ci in range(n_segs):
            chunk = x_chunks[:, ci, :, :]         # (B, seg, N)
            out_c, spk_c, mems = self._process_chunk(chunk, mems)
            all_outs.append(out_c)

            if not all_spks:
                all_spks = [[s] for s in spk_c]
            else:
                for i, s in enumerate(spk_c):
                    all_spks[i].append(s)

            # Detach state: TBPTT — gradients bounded to current chunk
            mems = [m.detach() for m in mems]

        # Reconstruct full sequence via the shared helper
        out_dim   = all_outs[0].shape[-1]
        out_stack = torch.stack(all_outs, dim=1)          # (B, n_segs, seg, out_dim)
        out_flat  = out_stack.reshape(B * n_segs, seg, out_dim)
        out_full  = reconstruct_sequence(
            out_flat, B, n_segs, seg, hop, L_pad, L)      # (B, out_dim, L)

        masks = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)

        # Flatten spike records: (B, total_t, hidden) → (B*total_t, hidden)
        flat_recs = []
        for layer_spks in all_spks:
            cat = torch.cat(layer_spks, dim=1)             # (B, n_segs*seg, hidden)
            flat_recs.append(cat.reshape(-1, cat.shape[-1]))

        return masks, flat_recs


# ─────────────────────────────────────────────────────────────────────────────
# StatefulGRUSeparator  [v11 — fast stateful GRU, drop-in for StatefulSNNSeparator]
# ─────────────────────────────────────────────────────────────────────────────

class StatefulGRUSeparator(nn.Module):
    """
    GRU separator with cross-chunk hidden state (v11 default).

    Same stateful design as StatefulSNNSeparator:
      - Non-overlapping chunks (hop = chunk_size)
      - Hidden state detached and carried between chunks (TBPTT)

    Key difference: a single cuDNN GRU kernel processes the full (B, seg, N)
    chunk instead of seg=100 Python LIF iterations.
    Speed vs StatefulSNNSeparator: ~100x faster inner loop.
    Expected epoch time: 2000-4000 s (vs ~35000 s for the LIF version).
    """

    def __init__(self, n_filters: int, hidden: int, n_layers: int,
                 n_speakers: int = 2, dropout: float = 0.1,
                 chunk_size: int = 100):
        super().__init__()
        self.n_speakers = n_speakers
        self.n_filters  = n_filters
        self.chunk_size = chunk_size
        self.hop_size   = chunk_size    # non-overlapping: each frame processed once

        self.in_norm  = nn.LayerNorm(n_filters)
        self.gru      = nn.GRU(
            input_size    = n_filters,
            hidden_size   = hidden,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = False,      # unidirectional: state can carry forward
        )
        self.out_proj = nn.Linear(hidden, n_filters * n_speakers)

    def forward(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, N, L = enc.shape
        seg = self.chunk_size
        hop = self.hop_size

        x_norm   = self.in_norm(enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_flat, _, n_segs, L_pad = segment_sequence(x_norm, seg, hop)
        x_chunks = x_flat.reshape(B, n_segs, seg, N)

        h_state: torch.Tensor = torch.zeros(
            self.gru.num_layers, B, self.gru.hidden_size,
            device=enc.device, dtype=enc.dtype,
        )
        all_outs: List[torch.Tensor] = []

        for ci in range(n_segs):
            chunk        = x_chunks[:, ci, :, :]           # (B, seg, N)
            out, h_state = self.gru(chunk, h_state)        # out: (B, seg, hidden)
            all_outs.append(self.out_proj(out))             # (B, seg, N*n_speakers)
            h_state      = h_state.detach()                 # TBPTT

        out_dim   = all_outs[0].shape[-1]
        out_stack = torch.stack(all_outs, dim=1)            # (B, n_segs, seg, out_dim)
        out_flat  = out_stack.reshape(B * n_segs, seg, out_dim)
        out_full  = reconstruct_sequence(out_flat, B, n_segs, seg, hop, L_pad, L)

        masks = out_full.view(B, self.n_speakers, N, L).softmax(dim=1)
        return masks, []    # no spike records


# ─────────────────────────────────────────────────────────────────────────────
# ConvDecoder  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
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
            blocks.extend([conv, nn.GroupNorm(n_groups, n_filters), nn.PReLU(n_filters)])
        self.refine = nn.Sequential(*blocks)
        _up = nn.ConvTranspose1d(n_filters, 1, kernel_size=kernel_sz, stride=stride, bias=False)
        self.upsample = _weight_norm(_up, name='weight', dim=0) if use_weight_norm else _up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.refine(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class SNNTasNet(nn.Module):
    """
    SNN-TasNet v10: stateful spiking separator + non-spiking ConvDecoder.

    snn_mode options:
      'gru'          → ChunkedGRUSeparator    (bidirectional GRU, pre-training)
      'snn'          → ChunkedSNNSeparator    (stateless LIF, v7 behaviour)
      'snn_stateful' → StatefulSNNSeparator   [NEW — v10 default]
    """

    def __init__(
        self,
        n_filters:       int   = 256,
        kernel_sz:       int   = 32,
        stride:          int   = 16,
        hidden:          int   = 512,    # v10 default: 512 (up from v7's 256)
        n_layers:        int   = 6,      # v10 default:   6 (up from v7's 3)
        n_speakers:      int   = 2,
        beta:            float = 0.9,
        dropout:         float = 0.35,
        snn_mode:        str   = "snn_stateful",
        chunk_size:      int   = 250,    # GRU mode only
        snn_chunk:       int   = 100,
        freq_mask:       int   = 27,
        n_freq_masks:    int   = 2,
        use_weight_norm: bool  = False,
        decoder_refine:  int   = 3,
        decoder_groups:  int   = 8,
    ):
        super().__init__()
        self.n_speakers      = n_speakers
        self.stride          = stride
        self.kernel_sz       = kernel_sz
        self.n_filters       = n_filters
        self.snn_mode        = snn_mode
        self.use_weight_norm = use_weight_norm

        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_sz, stride=stride, bias=False),
            nn.ReLU(),
        )
        self.spec_aug = SpecAugment(freq_mask_param=freq_mask, num_masks=n_freq_masks)

        if snn_mode in ("snn_stateful", "snn") and not HAS_SNN:
            print("[model] snntorch not installed — falling back to gru mode")
            snn_mode = "gru"

        if snn_mode == "snn_stateful":
            self.separator = StatefulSNNSeparator(
                n_filters, hidden, n_layers, n_speakers,
                beta, 1.0, dropout, snn_chunk,
            )
        elif snn_mode == "snn":
            self.separator = ChunkedSNNSeparator(
                n_filters, hidden, n_layers, n_speakers,
                beta, 1.0, dropout, snn_chunk,
            )
        elif snn_mode == "gru_stateful":
            self.separator = StatefulGRUSeparator(
                n_filters, hidden, n_layers, n_speakers, dropout, snn_chunk,
            )
        else:  # "gru"
            self.separator = ChunkedGRUSeparator(
                n_filters, hidden, n_layers, n_speakers, dropout, chunk_size
            )

        self.decoder = ConvDecoder(
            n_filters       = n_filters,
            kernel_sz       = kernel_sz,
            stride          = stride,
            n_refine        = decoder_refine,
            n_groups        = decoder_groups,
            use_weight_norm = use_weight_norm,
        )

    def forward(self, mixture: torch.Tensor) -> Dict:
        B, T   = mixture.shape
        enc    = self.encoder(mixture.unsqueeze(1))         # (B, N, L)
        enc    = self.spec_aug(enc)
        masks, spike_recs = self.separator(enc)             # (B, C, N, L)

        B_i, N_i, L_enc = enc.shape
        masked    = enc.unsqueeze(1) * masks                # (B, C, N, L)
        flat_in   = masked.reshape(B_i * self.n_speakers, N_i, L_enc)
        separated = self.decoder(flat_in).squeeze(1)        # (B*C, T')
        separated = separated.reshape(B_i, self.n_speakers, -1)

        T_out = separated.shape[-1]
        if T_out > T:
            separated = separated[..., :T]
        elif T_out < T:
            separated = F.pad(separated, (0, T - T_out))

        return {"separated": separated, "masks": masks, "spike_recs": spike_recs}


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer  (encoder + decoder only when separator dims change)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_encoder_decoder(src: SNNTasNet, dst: SNNTasNet,
                              verbose: bool = True) -> SNNTasNet:
    """Copy encoder and decoder from src to dst. Separator is left untouched."""
    assert src.n_filters == dst.n_filters
    assert src.kernel_sz == dst.kernel_sz

    dst.encoder.load_state_dict(src.encoder.state_dict())

    src_dec = src.decoder
    dst_dec = dst.decoder

    if isinstance(src_dec, ConvDecoder):
        try:
            dst_dec.load_state_dict(src_dec.state_dict())
            if verbose:
                print("[transfer] Decoder: full ConvDecoder → ConvDecoder copy")
        except RuntimeError as e:
            if verbose:
                print(f"[transfer] Decoder copy failed ({e}); refine blocks kept fresh")
    else:
        try:
            dst_dec.upsample.load_state_dict(src_dec.state_dict())
            if verbose:
                print("[transfer] Decoder: ConvTranspose1d → upsample (refine: fresh)")
        except RuntimeError:
            try:
                src_w = src_dec.weight.data
            except AttributeError:
                g = src_dec.weight_g.data
                v = src_dec.weight_v.data
                norms = v.view(v.shape[0], -1).norm(dim=1, keepdim=True).view(-1, 1, 1)
                src_w = g * v / norms.clamp(min=1e-8)
            dst_up = dst_dec.upsample
            if hasattr(dst_up, 'weight_v'):
                norms = src_w.view(src_w.shape[0], -1).norm(dim=1)
                dst_up.weight_v.data.copy_(src_w)
                dst_up.weight_g.data.copy_(norms.view(-1, 1, 1))
            else:
                dst_up.weight.data.copy_(src_w)
            if verbose:
                print("[transfer] Decoder: weight transferred to upsample (refine: fresh)")

    if verbose:
        print(f"[transfer] Encoder: {count_parameters(src.encoder):,} params copied")
        print(f"[transfer] Decoder: {count_parameters(dst.decoder):,} params copied")
        print(f"[transfer] Separator: freshly initialised "
              f"({count_parameters(dst.separator):,} params)")

    return dst


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    B, T = 2, 64000
    mix  = torch.randn(B, T)

    print("=== v10 stateful SNN mode ===")
    if HAS_SNN:
        model = SNNTasNet(snn_mode="snn_stateful", hidden=512, n_layers=6)
        n_params = count_parameters(model)
        print(f"  encoder:   {count_parameters(model.encoder):>10,} params")
        print(f"  separator: {count_parameters(model.separator):>10,} params  "
              f"(hidden=512, n_layers=6, stateful)")
        print(f"  decoder:   {count_parameters(model.decoder):>10,} params")
        print(f"  total:     {n_params:>10,} params")
        model.train()
        t0  = time.time()
        out = model(mix)
        print(f"  forward:  {time.time()-t0:.2f}s  out={out['separated'].shape}  "
              f"spike_recs={len(out['spike_recs'])}")
        if out['spike_recs']:
            rates = [r.float().mean().item() for r in out['spike_recs']]
            print(f"  spike rates per layer: {[f'{r:.3f}' for r in rates]}")
    else:
        print("  [!] snntorch not installed. Run: pip install snntorch")

    print("\n=== v10 GRU mode (reference) ===")
    gru = SNNTasNet(snn_mode="gru", hidden=512, n_layers=3)
    print(f"  total: {count_parameters(gru):,} params")
    gru.train()
    out = gru(mix)
    print(f"  forward: out={out['separated'].shape}")
