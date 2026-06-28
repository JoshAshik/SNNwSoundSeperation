# SNN-TasNet: Spiking Neural Network Audio Source Separation

An audio source separation system built on the TasNet framework that separates a mono mixture of two
speakers into individual waveforms. The project began as a **Spiking Neural Network (SNN)** separator
using Leaky Integrate-and-Fire (LIF) neurons (v1–v9) and evolved to a **Dual-Path RNN (DPRNN)**
separator (v13+) once the SNN separator's compute cost and accuracy ceiling were established. The
encoder/decoder and PIT/loss machinery are shared across both.

## Architecture (v17 — active)

```
Input: mono mixture (16 kHz, 4s = 64,000 samples)

Encoder:     Conv1d (1 → 256, kernel=16, stride=8) + ReLU        [finer front-end, v17]
             → ~8000 frames (≈2× the earlier kernel=32/stride=16 encoder)

Separator:   DPRNNSeparator (v13+, active)
             - bottleneck to bn_dim=64, unfold into chunks of 200 (50% overlap)
             - 6× dual-path blocks: intra-chunk BiLSTM (fine temporal) +
               inter-chunk BiLSTM (full-sequence context), rnn_hidden=128
             - softmax masks across the speaker dim (energy conservation)

Decoder:     ConvDecoder — 3× Conv1d refinement + ConvTranspose1d (k=16, s=8) upsample

Loss:        pure SI-SDR (PIT) + 0.1·spectral     [v17 — metric-aligned objective]
Training:    end-to-end from scratch, Adam lr=1e-3 + ReduceLROnPlateau

Output: 2 separated waveforms
```

> **v17 hypothesis:** v13–v16 plateaued at a ~0 dB train-val gap (underfitting). v17 attacks the
> front-end directly — a finer encoder stride, a pure SI-SDR loss, and a from-scratch training recipe —
> rather than adding data augmentation. `oracle_mask_diagnostic.py` bounds the finer encoder's headroom
> before training. The DPRNN separator (v13) is still the active separator; the LIF/SNN separators below
> are retained for reference.

### LIF Neuron Dynamics (used in SNN separator variants)

```
U[t] = β·U[t-1] + I[t]     # membrane potential integration
S[t] = Θ(U[t] - θ)          # spike firing (Heaviside)
U[t] ← U[t]·(1 - S[t])     # post-spike reset
```

- **β** (decay): learnable, clamped to [0.5, 0.99] to prevent saturation
- **θ** (threshold): learnable
- Surrogate gradient: fast-sigmoid (slope=25) via [snntorch](https://github.com/jeshraghian/snntorch)

## Project Evolution

The project evolved through two tracks: an 8-channel fixed-category separator (v1–v2) and a 2-speaker permutation-invariant separator (v3–v17, active).

### 8-Channel Track (v1–v2) — Shelved

Fixed output channels for 8 sound categories (voice, animal, impacts, music, weather, alerts, domestic, tools). Trained on FSD50K. Peak: **+0.15 dB SI-SDRi**. Shelved due to channel collapse and spike saturation.

### 2-Speaker Track (v3–v17) — Active

| Version | Dataset | Architecture | Val SI-SDRi | Key Change |
|---------|---------|-------------|-------------|------------|
| v3 | FSD50K | ChunkedSNN (stateless LIF) | +3.83 dB | Permutation Invariant Training (PIT) |
| v4 | FSD50K | ChunkedSNN | +3.76 dB | Softmax masks, per-layer spike penalty |
| v5 | FSD50K | ChunkedSNN | +3.73 dB | Weight norm, RMS recon loss |
| v6 | FSD50K | ChunkedSNN | +3.82 dB | Balance loss — hit FSD50K ceiling |
| v7 | Libri2Mix | ChunkedSNN + ConvDecoder | -1.16 dB | Pivoted to speech; multi-layer decoder |
| v8 | Libri2Mix | ChunkedSNN + ConvDecoder | -1.65 dB | Dropout 0.5 over-regularised |
| v9 | Libri2Mix | ChunkedSNN + ConvDecoder | -2.48 dB | Double-applied gain augmentation |
| v10 | Libri2Mix | StatefulSNN | CANCELLED | 4000 CUDA dispatches/fwd — too slow |
| v11 | Libri2Mix | StatefulGRU | +5.15 dB | cuDNN GRU with cross-chunk state |
| v12 | Libri2Mix | StatefulGRU | +5.46 dB | Stage-2 fine-tune; augmentation neutral |
| **v13** | **Libri2Mix** | **DPRNN** | **+7.22 dB** | **Dual-path RNN separator (+1.76 dB)** |
| v14 | Libri2Mix | DPRNN (wider) | +7.21 dB | rnn_hidden 128→256 — no gain; data-bound |
| v15 | Libri2Mix (dynamic) | DPRNN | +6.64 dB | Dynamic mixing + full augment — over-regularised |
| v16 | Libri2Mix (dynamic) | DPRNN | +6.79 dB | Dynamic mixing only — still < v13 (diverges from eval) |
| **v17** | **Libri2Mix (fixed)** | **DPRNN, finer encoder** | **In progress** | **k=16/s=8 from scratch + pure SI-SDR loss** |

**Best so far:** +7.22 dB (v13). **Target:** val SI-SDRi > **+10 dB** (Conv-TasNet baseline on Libri2Mix: ~14 dB)

### Key Architectural Decisions

- **FSD50K → Libri2Mix (v7):** FSD50K environmental sounds hit a quality ceiling at ~3.8 dB; pivoted to Libri2Mix clean speech for fair comparison with literature
- **ConvDecoder (v7):** Single ConvTranspose1d decoder caused hissing artifacts; replaced with 3-layer Conv1d refinement stack
- **Stateful separator (v10→v11):** Stateless LIF reset membrane state every 0.1s chunk, forcing speaker re-identification; stateful variants carry hidden state across chunks
- **GRU over SNN separator (v11):** StatefulSNNSeparator required 4000 Python-CUDA dispatches per forward pass (~20s/batch); StatefulGRUSeparator uses a single cuDNN kernel per chunk
- **DPRNN separator (v13):** GRU separator hit a ~+5.5 dB capacity ceiling (train and val both capped → not overfitting). Dual-path RNN alternates intra-chunk and inter-chunk BiLSTMs; +1.76 dB over v12 to +7.22 dB
- **Finer encoder, from scratch (v17):** v13–v16 plateaued with a ~0 dB train-val gap (underfitting). The coarse kernel=32/stride=16 front-end is the suspected ceiling; v17 halves both (k=16, s=8), drops the scale-dependent reconstruction loss for pure SI-SDR, and trains end-to-end from scratch with a plateau LR schedule. An oracle-mask diagnostic gates the run

## File Structure

### Active Pipeline (v17)

| File | Purpose |
|------|---------|
| [`sep_model_v10.py`](sep_model_v10.py) | Model architecture — DPRNNSeparator (active) + StatefulGRU/SNN separators, ConvDecoder, SNNTasNet |
| [`two_speaker_train_v17.py`](two_speaker_train_v17.py) | Training loop — finer encoder (k=16/s=8) from scratch, pure SI-SDR loss, Adam + ReduceLROnPlateau |
| [`oracle_mask_diagnostic.py`](oracle_mask_diagnostic.py) | STFT oracle-mask (IRM/IBM) SI-SDRi ceiling per front-end window — v17 gate |
| [`librimix_dataset.py`](librimix_dataset.py) | Libri2Mix dataset loader (fixed real mixtures; augmentation optional) |
| [`eval_dataset.py`](eval_dataset.py) | Batch SI-SDRi evaluation on dev/test splits |
| [`two_speaker_inference.py`](two_speaker_inference.py) | Inference and SI-SDRi evaluation |
| [`slurm_v17_twospeaker.sh`](slurm_v17_twospeaker.sh) | Self-resubmitting SLURM job for ARC cluster |
| [`smoke_test_v17.py`](smoke_test_v17.py) | End-to-end smoke test (synthetic data → train/resume/oracle) |

### Reference Versions

| File | Version | Notes |
|------|---------|-------|
| [`two_speaker_train_v15.py`](two_speaker_train_v15.py) | v15/v16 | Dynamic mixing + speed perturbation (DPRNN) — dropped in v17 |
| [`dynamic_mix_dataset.py`](dynamic_mix_dataset.py) | v15/v16 | On-the-fly clean speech mixing — diverged from eval distribution |
| [`two_speaker_train_v14.py`](two_speaker_train_v14.py) | v14 | Wider DPRNN (rnn_hidden=256) — no improvement |
| [`two_speaker_train_v13.py`](two_speaker_train_v13.py) | v13 | DPRNN separator (rnn_hidden=128) — +7.22 dB best |
| [`two_speaker_train_v12.py`](two_speaker_train_v12.py) | v12 | Stage-2 GRU fine-tune (warmstart, ablation) |
| [`two_speaker_train_v11.py`](two_speaker_train_v11.py) | v11 | StatefulGRU training (PIT loss, EMA) |
| [`sep_model_v7.py`](sep_model_v7.py) | v7 | Stateless ChunkedSNN (hidden=256, 3 layers) |
| [`two_speaker_train_v7.py`](two_speaker_train_v7.py) | v7 | Training with ChunkedSNN separator |
| [`sep_model.py`](sep_model.py) | v3–v6 | Original architecture with single-layer decoder |
| [`two_speaker_train.py`](two_speaker_train.py) | v3–v6 | FSD50K training loop |
| [`two_speaker_dataset.py`](two_speaker_dataset.py) | v3–v6 | On-the-fly FSD50K 2-speaker mixer |

### 8-Channel Legacy

| File | Purpose |
|------|---------|
| [`snn_finetune.py`](snn_finetune.py) | 8-channel training with FSD50K |
| [`snn_finetune_v2.py`](snn_finetune_v2.py) | v2 training (beta clamp fix) |
| [`fsd50k_dataset.py`](fsd50k_dataset.py) | FSD50K loader and scene synthesiser |
| [`sep_inference.py`](sep_inference.py) | 8-channel inference |
| [`wham_pretrain.py`](wham_pretrain.py) | GRU pre-training on MiniLibriMix |

### Shared Utilities

| File | Purpose |
|------|---------|
| [`sep_losses.py`](sep_losses.py) | Loss functions (SI-SDR, spectral, spike rate) |
| [`sep_metrics.py`](sep_metrics.py) | SI-SDRi metric computation |
| [`config.py`](config.py) | Label constants for 8-channel categories |
| [`encoding.py`](encoding.py) | Audio encoding utilities |

## Documentation

Detailed documentation in [`docs/`](docs/):

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Full model graphs for all versions, LIF dynamics, loss functions, configs |
| [Training](docs/training.md) | Training status, hyperparameters, full history, commands |
| [Dataset](docs/dataset.md) | Libri2Mix (active) and FSD50K (legacy) details |
| [Environment](docs/environment.md) | ARC cluster setup, local Windows environment |
| [Bugs](docs/bugs.md) | 30+ diagnosed and fixed bugs |
| [Decisions](docs/decisions.md) | Design rationale for key choices |
| [Inference](docs/inference.md) | Inference scripts, flags, and checkpoint compatibility |

## Setup

```bash
pip install torch torchaudio snntorch tensorboard pandas numpy mir_eval pesq
```

## Training (ARC Cluster)

```bash
# Deploy and run on NCSU ARC
scp sep_model_v10.py two_speaker_train_v17.py slurm_v17_twospeaker.sh oracle_mask_diagnostic.py \
    librimix_dataset.py eval_dataset.py user@arc.csc.ncsu.edu:~/SNNwSoundSeperation/
ssh arc.csc.ncsu.edu
cd ~/SNNwSoundSeperation
sed -i 's/\r//' slurm_v17_twospeaker.sh

# Gate first: confirm the finer encoder has headroom on dev (use the conda python)
/path/to/conda/envs/snn/bin/python oracle_mask_diagnostic.py \
    --librimix_root /path/to/Libri2Mix/wav16k/max --split dev --max_samples 300

# Then launch (self-resubmits until 200 epochs complete)
mkdir -p checkpoints_v17 && sbatch slurm_v17_twospeaker.sh
```

## Evaluation & Inference

```bash
# Batch SI-SDRi over a full split
python3 eval_dataset.py --model checkpoints_v17/best_2spk.pt \
    --librimix_root /path/to/Libri2Mix/wav16k/max --split dev

# Per-sample inference with audio output (checkpoint arch read from model_cfg — v11–v17 compatible)
python3 two_speaker_inference.py \
    --checkpoint checkpoints_v17/best_2spk.pt \
    --from_dataset \
    --librimix_root /path/to/Libri2Mix/wav16k/max \
    --n_samples 10
```
