# Inference
<!-- dependencies: two_speaker_inference.py, sep_inference.py, sep_model.py -->

## Two-speaker inference (two_speaker_inference.py)

### Basic usage
```bash
# Separate a mixture into two waveforms:
python3 two_speaker_inference.py \
    --model  checkpoints_snn/best_snn_v2.pt \
    --input  mix.wav \
    --out_dir ./pit_output

# With reference sources for SI-SDRi scoring:
python3 two_speaker_inference.py \
    --model  checkpoints_snn/best_snn_v2.pt \
    --input  mix.wav \
    --ref1   source1.wav \
    --ref2   source2.wav \
    --out_dir ./pit_output

# Also save a copy of the input mixture:
python3 two_speaker_inference.py \
    --model  checkpoints_snn/best_snn_v2.pt \
    --input  mix.wav \
    --save_mixture
```

### All flags
| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to `.pt` checkpoint |
| `--input` | required | Path to input mixture WAV |
| `--out_dir` | `./pit_output` | Directory for output files |
| `--ref1` | None | Reference WAV for speaker 1 (enables SI-SDRi) |
| `--ref2` | None | Reference WAV for speaker 2 (enables SI-SDRi) |
| `--save_mixture` | off | Save input mixture to out_dir as `mixture.wav` |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--max_seconds` | `30.0` | Truncate input to avoid OOM on long files |

### Outputs
```
pit_output/
├── speaker_1_est.wav   — separated channel 0, peak-normalised
└── speaker_2_est.wav   — separated channel 1, peak-normalised
```

### Diagnostics printed to console
- Per-channel output RMS (active vs silent)
- SI-SDRi with best-permutation assignment (if `--ref1`/`--ref2` provided)
- Spike rate per LIF layer (target ~0.10; > 0.25 flagged as saturated)
- Mask statistics: per-channel mean and saturation % (binary = decided; soft = uncertain)
- Reconstruction error rel-L1 (target < 0.10; high = energy not conserved)

---

## How the forward pass works (v11 — current)

```
Mixture (1, T)
  → peak_norm                       → (1, T)  in [-1, 1]
  → SNNTasNet.forward()  [sep_model_v10.py, snn_mode="gru_stateful"]
      Encoder:    Conv1d(1→256, k=32, s=16)     → (1, 256, 4000)
      Separator:  StatefulGRUSeparator
                    40 non-overlapping chunks, GRU hidden state carries between chunks
                    → softmax masks (1, 2, 256, 4000)
      masked      enc.unsqueeze(1) * masks       → (1, 2, 256, 4000)
      Decoder:    ConvDecoder
                    3× Conv1d(256,256,k=3)+GroupNorm+PReLU
                    ConvTranspose1d(256→1, k=32, s=16) → (2, 1, T')
                    reshape → (1, 2, T')
  → Trim T' → T
  → output["separated"] (1, 2, T), output["masks"], output["spike_recs"]=[]
```

**v7** used ChunkedSNNSeparator — spike_recs is a non-empty list. Spike rate diagnostics
printed by `two_speaker_inference.py` are only meaningful for v7-era checkpoints.

EMA weights (`ema_state["shadow"]`) are always used for inference when present.

**v6 and earlier** used a single `ConvTranspose1d` decoder (no refinement blocks).
`two_speaker_inference.py` loads the model architecture from `model_cfg` in the checkpoint,
so it works with both v6/v7 (`sep_model.py`/`sep_model_v7.py`) and v10/v11 (`sep_model_v10.py`)
checkpoints provided the correct model module is imported.

---

## 8-channel inference (sep_inference.py)

For the existing v1/v2 8-channel checkpoints:

```bash
# Separate a single WAV (outputs one file per active sound category):
python3 sep_inference.py \
    --checkpoint checkpoints_snn/best_snn_v2.pt \
    --input      my_mix.wav \
    --output_dir ./sep_output

# Pull random scenes from the original 400-scene dataset:
python3 sep_inference.py \
    --checkpoint checkpoints_snn/best_snn_v2.pt \
    --from_dataset \
    --data_root  ./data \
    --n_samples  10
```

Silent output channels (RMS < 1e-3) are skipped automatically.

---

## Checkpoint compatibility

Both inference scripts read `model_cfg` from the checkpoint and build the
model automatically. No manual arch flags needed.

| Script | n_speakers | Checkpoint |
|---|---|---|
| `two_speaker_inference.py` | 2 | Trained with `n_speakers=2` |
| `sep_inference.py` | 8 | `best_snn_v2.pt`, `best_snn_v1_gru_ep696.pt` |

Loading an 8-speaker checkpoint in `two_speaker_inference.py` will print a
warning and proceed — only `speaker_1_est.wav` and `speaker_2_est.wav` (channels
0 and 1) will be saved. The other 6 channels are silently dropped.
