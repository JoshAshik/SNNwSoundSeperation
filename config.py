"""
Configuration for SNN Sound Source Separation.
Dataset: 400 binaural scenes, 3 s each, sourced from the provided metadata.

Architecture philosophy:
  With only ~300 training samples, the model must be tiny to avoid overfitting.
  Target: ~5,000-15,000 trainable parameters total.
  Rule of thumb: no more than ~50 params per training sample.
"""

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────
# Label spaces (from metadata.json)
# ──────────────────────────────────────────────
SOURCE_LABELS     = ["human_voice", "animal", "impacts", "music", "weather", "alerts"]
TRANSIENT_LABELS  = ["domestic_and_objects", "tools_and_machines"]
DYNAMIC_LABELS    = ["sbs_x1.0_y-1.0"]

ALL_LABELS = SOURCE_LABELS + TRANSIENT_LABELS   # 8-class multi-label target

# ──────────────────────────────────────────────
# Audio / feature settings
# ──────────────────────────────────────────────
@dataclass
class AudioConfig:
    sample_rate:    int   = 16_000
    duration_ms:    int   = 3_000
    n_samples:      int   = 48_000
    channels:       int   = 2

@dataclass
class CochleagramConfig:
    n_filters:      int   = 64
    fmin:           float = 50.0
    fmax:           float = 8_000.0
    frame_ms:       float = 10.0
    hop_ms:         float = 5.0

    @property
    def n_frames(self) -> int:
        hop_samples = int(AudioConfig().sample_rate * self.hop_ms / 1000)
        return AudioConfig().n_samples // hop_samples  # ~600

# ──────────────────────────────────────────────
# Spike encoding
# ──────────────────────────────────────────────
@dataclass
class EncoderConfig:
    T:              int   = 100
    max_rate:       float = 100.0
    threshold:      float = 0.1

# ──────────────────────────────────────────────
# SNN model  -- DRASTICALLY REDUCED for 400-sample dataset
# ──────────────────────────────────────────────
@dataclass
class SNNConfig:
    # LIF neuron parameters
    beta:           float = 0.9
    threshold:      float = 1.0

    # Architecture: 256 -> 64 -> 32  (~18,000 params total vs old 640,000)
    input_size:     int   = 64 * 4      # 256
    hidden_sizes:   List[int] = field(default_factory=lambda: [128, 64])
    n_classes:      int   = len(ALL_LABELS)   # 8

    # Slot attention -- simplified
    n_source_slots: int   = 2           # was 3
    separation_dim: int   = 32          # was 128

    dropout:        float = 0.4

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    data_root:      str   = "./data"
    labels_csv:     str   = "./labels.csv"
    data_map_json:  str   = "./original_data_map.json"

    n_epochs:       int   = 200
    batch_size:     int   = 16
    lr:             float = 1e-3
    weight_decay:   float = 5e-3        # strong L2 for tiny dataset
    val_split:      float = 0.15
    test_split:     float = 0.10
    seed:           int   = 42

    # Loss weights
    lambda_ce:      float = 1.0
    lambda_sep:     float = 0.0         # disabled -- hurts more than helps here

    device:         str   = "cuda"
    num_workers:    int   = 4
    checkpoint_dir: str   = "./checkpoints"
    log_dir:        str   = "./runs"

# ──────────────────────────────────────────────
# Instantiated defaults (import these)
# ──────────────────────────────────────────────
audio_cfg   = AudioConfig()
cochlea_cfg = CochleagramConfig()
enc_cfg     = EncoderConfig()
snn_cfg     = SNNConfig()
train_cfg   = TrainConfig()