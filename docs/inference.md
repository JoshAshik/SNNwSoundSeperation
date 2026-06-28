# Inference
<!-- dependencies: two_speaker_inference.py, eval_dataset.py, sep_inference.py, sep_model_v10.py, sep_model.py -->

## Two-speaker inference (two_speaker_inference.py)

### Basic usage
```bash
# Separate a mixture into two waveforms (v11/v12 checkpoint):
python3 two_speaker_inference.py \
    --model  checkpoints_v11/best_2spk.pt \
    --input  mix.wav \
    --out_dir ./pit_output

# With reference sources for SI-SDRi scoring:
python3 two_speaker_inference.py \
    --model  checkpoints_v12/best_2spk.pt \
    --input  mix.wav \
    --ref1   source1.wav \
    --ref2   source2.wav \
    --out_dir ./pit_output

# Also save a copy of the input mixture:
python3 two_speaker_inference.py \
    --model  checkpoints_v12/best_2spk.pt \
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
so it works with both v6/v7 (`sep_model.py`/`sep_model_v7.py`) and v10–v17 (`sep_model_v10.py`)
checkpoints. The import uses a try/except fallback: prefers `sep_model_v10` (v11+), falls back
to `sep_model` (v3-v7) if `sep_model_v10.py` is not present.

**v17 note:** v17 checkpoints carry a finer encoder (`kernel_sz=16, stride=8`) in `model_cfg`.
Because both inference scripts read `kernel_sz`/`stride` from `model_cfg` and rebuild the encoder
and `ConvTranspose1d` decoder accordingly, no flags or code changes are needed — the finer front-end
loads automatically alongside the matching v13-sized DPRNN separator (`dprnn_bn_dim`, `dprnn_rnn_hidden`).

---

## Batch evaluation (eval_dataset.py)

Evaluates a checkpoint on an entire Libri2Mix split (dev, test, or train-100).
Reports per-sample SI-SDRi with summary statistics (mean, median, std, worst-5).

### Usage
```bash
# Full dev evaluation:
python3 eval_dataset.py --model checkpoints_v12/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --output_csv eval_dev_v12.csv

# Fast dev50 subset (for iteration during ablation):
python3 eval_dataset.py --model checkpoints_v12/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --max_samples 50
```

### Flags
| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to `.pt` checkpoint |
| `--librimix_root` | required | Libri2Mix wav16k/max root |
| `--split` | `dev` | `dev`, `test`, or `train-100` |
| `--max_samples` | `0` (all) | Limit to first N samples (use 50 for fast iteration) |
| `--output_csv` | none | Save per-sample SI-SDRi to CSV |

### Output
- Running mean SI-SDRi printed every 50 samples
- Final summary: mean, median, std, min, max, worst-5 samples
- Optional CSV with columns: `stem`, `si_sdri_db`

Uses EMA-shadow weights when present, same as single-file inference.
Uses PIT (best permutation) for each sample independently.

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
| `two_speaker_inference.py` | 2 | v11–v17 checkpoints (trained with `n_speakers=2`; v17 finer encoder auto-detected) |
| `sep_inference.py` | 8 | `best_snn_v2.pt`, `best_snn_v1_gru_ep696.pt` |

Loading an 8-speaker checkpoint in `two_speaker_inference.py` will print a
warning and proceed — only `speaker_1_est.wav` and `speaker_2_est.wav` (channels
0 and 1) will be saved. The other 6 channels are silently dropped.
