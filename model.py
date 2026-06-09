"""
model.py -- Spiking Neural Network for binaural sound source separation.

REDESIGNED for small dataset (400 samples):
  - Hidden sizes: 256->64->32 instead of 256->512->256->128
  - No slot attention (replaced with simple global average pool)
  - Total params: ~18,000 vs previous ~640,000
  - Strong dropout (0.5) throughout

Requires:  snntorch >= 0.9
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False
    print("[model] WARNING: snntorch not found.")

from config import SNNConfig, EncoderConfig, snn_cfg, enc_cfg, ALL_LABELS


# ──────────────────────────────────────────────────────────────────────────────
# Surrogate gradient
# ──────────────────────────────────────────────────────────────────────────────

def get_spike_grad():
    if HAS_SNNTORCH:
        return surrogate.fast_sigmoid(slope=25)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# LIF layer
# ──────────────────────────────────────────────────────────────────────────────

class LIFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: SNNConfig = snn_cfg,
                 dropout: float = 0.0):
        super().__init__()
        self.fc   = nn.Linear(in_features, out_features, bias=True)
        self.bn   = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(p=dropout)
        self._out = out_features

        if HAS_SNNTORCH:
            self.lif = snn.Leaky(
                beta=cfg.beta,
                threshold=cfg.threshold,
                spike_grad=get_spike_grad(),
                learn_beta=True,
                learn_threshold=True,
            )
        else:
            self.lif = None

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self._out, device=device)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.fc(x)
        cur = self.bn(cur)
        cur = self.drop(cur)

        if HAS_SNNTORCH:
            spk, mem_next = self.lif(cur, mem)
        else:
            mem_next = 0.9 * mem + cur
            spk      = (mem_next >= 1.0).float()
            mem_next = mem_next * (1 - spk)

        return spk, mem_next


# ──────────────────────────────────────────────────────────────────────────────
# Compact SNN Encoder (no recurrence -- saves ~260k params)
# ──────────────────────────────────────────────────────────────────────────────

class SNNEncoder(nn.Module):
    """
    Stacked LIF layers without recurrence.
    Input:   (T, B, D_in)
    Returns: spike_recs (list), readout (B, D_last)
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], cfg: SNNConfig = snn_cfg):
        super().__init__()
        sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList([
            LIFLayer(
                in_features=sizes[i],
                out_features=sizes[i+1],
                cfg=cfg,
                dropout=cfg.dropout if i < len(hidden_sizes) - 1 else 0.0,
            )
            for i in range(len(hidden_sizes))
        ])

    def forward(self, spikes: torch.Tensor) -> Tuple[List, torch.Tensor]:
        T, B, _ = spikes.shape
        device  = spikes.device

        mems      = [layer.init_hidden(B, device) for layer in self.layers]
        spike_recs = [[] for _ in self.layers]

        x = spikes
        for t in range(T):
            inp = x[t]
            for i, layer in enumerate(self.layers):
                spk, mems[i] = layer(inp, mems[i])
                spike_recs[i].append(spk)
                inp = spk

        spike_recs = [torch.stack(rec, dim=0) for rec in spike_recs]  # (T, B, D)
        readout    = spike_recs[-1].mean(dim=0)                        # (B, D_last)

        return spike_recs, readout


# ──────────────────────────────────────────────────────────────────────────────
# Simple Classification Head (replaces complex slot attention)
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Simple MLP head on top of the SNN readout.
    No slot attention -- that complexity is wasteful for 300 training samples.
    Uses prior-based bias init to prevent sigmoid saturation.
    """
    def __init__(self, cfg: SNNConfig = snn_cfg):
        super().__init__()
        hidden = cfg.hidden_sizes[-1]  # 32
        n_cls  = cfg.n_classes         # 8

        mid = max(hidden // 2, n_cls * 2)   # intermediate size
        self.head = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(mid, n_cls),
        )

        # Initialise output bias to log-odds of approximate class priors
        # Prevents all sigmoids saturating at 0.5 from random init
        # Priors estimated from dataset label distribution
        approx_priors = [0.45, 0.55, 0.45, 0.10, 0.35, 0.55, 0.70, 0.70]
        bias_inits    = [math.log(p / (1.0 - p + 1e-8)) for p in approx_priors]
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor(bias_inits, dtype=torch.float32))
            # Small weight init on output layer -- priors handle the offset,
            # small weights mean output starts near priors and grows gradually
            nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, readout: torch.Tensor) -> torch.Tensor:
        return self.head(readout)  # (B, n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class SNNSoundSeparator(nn.Module):
    """
    Compact end-to-end SNN model for multi-label sound classification.
    Designed for small datasets (~300 training samples).
    """
    def __init__(self, cfg: SNNConfig = snn_cfg):
        super().__init__()
        self.encoder  = SNNEncoder(cfg.input_size, cfg.hidden_sizes, cfg)
        self.head     = ClassificationHead(cfg)

    def forward(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        spike_recs, readout = self.encoder(spikes)
        logits              = self.head(readout)

        return {
            "logits":          logits,
            "slot_embeddings": readout.unsqueeze(1),  # dummy (B,1,D) for loss compat
            "spike_recs":      spike_recs,
            "mem_recs":        [],
            "readout":         readout,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class SeparationLoss(nn.Module):
    """
    BCE with optional pos_weight (capped at 4x) + spike rate regularisation.
    Slot orthogonality loss removed (no slots in compact model).
    """
    def __init__(
        self,
        lambda_ce:   float = 1.0,
        lambda_sep:  float = 0.0,   # unused, kept for API compatibility
        lambda_rate: float = 0.01,
        target_rate: float = 0.1,
        pos_weight:  "torch.Tensor | None" = None,
    ):
        super().__init__()
        self.lambda_ce   = lambda_ce
        self.lambda_rate = lambda_rate
        self.target_rate = target_rate

        if pos_weight is not None:
            pos_weight = pos_weight.clamp(max=4.0)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def rate_regularisation(self, spike_recs: List[torch.Tensor]) -> torch.Tensor:
        if not spike_recs:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=spike_recs[0].device)
        for rec in spike_recs:
            rate  = rec.mean()
            total = total + (rate - self.target_rate) ** 2
        return total / len(spike_recs)

    def forward(
        self,
        logits:     torch.Tensor,
        labels:     torch.Tensor,
        slots:      torch.Tensor,       # unused in compact model
        spike_recs: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ce_loss   = self.bce(logits, labels)
        rate_loss = self.rate_regularisation(spike_recs)
        sep_loss  = torch.tensor(0.0, device=logits.device)

        total = self.lambda_ce * ce_loss + self.lambda_rate * rate_loss

        return {"total": total, "ce": ce_loss, "sep": sep_loss, "rate": rate_loss}


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import snn_cfg, enc_cfg

    T, B, D = enc_cfg.T, 4, snn_cfg.input_size
    dummy   = torch.randint(0, 2, (T, B, D)).float()
    model   = SNNSoundSeparator(snn_cfg)
    print(f"Parameters: {count_parameters(model):,}")
    out     = model(dummy)
    print(f"Logits: {out['logits'].shape}")
    labels  = torch.zeros(B, snn_cfg.n_classes)
    labels[:, [2, 4]] = 1.0
    loss_fn = SeparationLoss()
    losses  = loss_fn(out["logits"], labels, out["slot_embeddings"], out["spike_recs"])
    print(f"Losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")