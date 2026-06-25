# SNN-TasNet: Spiking Neural Network Audio Source Separation

A biologically-inspired audio source separation system using **Spiking Neural Networks (SNNs)** built on the TasNet framework. The model separates a mono mixture of two speakers into individual waveforms using Leaky Integrate-and-Fire (LIF) neurons with learnable dynamics.

## Architecture

```
Input: mono mixture (16 kHz)

Encoder:     Conv1d (1 → 256, kernel=32, stride=16) + ReLU
             SpecAugment during training

Separator:   StatefulGRUSeparator (v11, active)
             - 40 non-overlapping chunks of 100 frames
             - GRU (hidden=512, 6 layers) with cross-chunk state
             - Softmax masks ensure energy conservation

Decoder:     ConvDecoder — 3× Conv1d refinement + ConvTranspose1d upsample

Output: 2 separated waveforms
```

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

The project evolved through two tracks: an 8-channel fixed-category separator (v1–v2) and a 2-speaker permutation-invariant separator (v3–v12).

### 8-Channel Track (v1–v2) — Shelved

Fixed output channels for 8 sound categories (voice, animal, impacts, music, weather, alerts, domestic, tools). Trained on FSD50K. Peak: **+0.15 dB SI-SDRi**. Shelved due to channel collapse and spike saturation.

### 2-Speaker Track (v3–v12) — Active

| Version | Dataset | Architecture | Val SI-SDRi | Key Change |
|---------|---------|-------------|-------------|------------|
| v3 | FSD50K | ChunkedSNN (stateless LIF) | **+3.83 dB** | Permutation Invariant Training (PIT) |
| v4 | FSD50K | ChunkedSNN | +3.76 dB | Softmax masks, per-layer spike penalty |
| v5 | FSD50K | ChunkedSNN | +3.73 dB | Weight norm, RMS recon loss |
| v6 | FSD50K | ChunkedSNN | +3.82 dB | Balance loss — hit FSD50K ceiling |
| v7 | Libri2Mix | ChunkedSNN + ConvDecoder | -1.16 dB | Pivoted to speech; multi-layer decoder |
| v8 | Libri2Mix | ChunkedSNN + ConvDecoder | -1.65 dB | Dropout 0.5 over-regularised |
| v9 | Libri2Mix | ChunkedSNN + ConvDecoder | -2.48 dB | Double-applied gain augmentation |
| v10 | Libri2Mix | StatefulSNN | CANCELLED | 4000 CUDA dispatches/fwd — too slow |
| v11 | Libri2Mix | StatefulGRU | +5.15 dB | cuDNN GRU with cross-chunk state |
| **v12** | **Libri2Mix** | **StatefulGRU** | **+5.46 dB** | **Stage-2 warmstart from v11; best Run A** |

**Target:** val SI-SDRi > **+10 dB** (Conv-TasNet baseline on Libri2Mix: ~14 dB)

### Key Architectural Decisions

- **FSD50K → Libri2Mix (v7):** FSD50K environmental sounds hit a quality ceiling at ~3.8 dB; pivoted to Libri2Mix clean speech for fair comparison with literature
- **ConvDecoder (v7):** Single ConvTranspose1d decoder caused hissing artifacts; replaced with 3-layer Conv1d refinement stack
- **Stateful separator (v10→v11):** Stateless LIF reset membrane state every 0.1s chunk, forcing speaker re-identification; stateful variants carry hidden state across chunks
- **GRU over SNN separator (v11):** StatefulSNNSeparator required 4000 Python-CUDA dispatches per forward pass (~20s/batch); StatefulGRUSeparator uses a single cuDNN kernel per chunk

## File Structure

### Active Pipeline (v12)

| File | Purpose |
|------|---------|
| [`sep_model_v10.py`](sep_model_v10.py) | Model architecture — StatefulGRUSeparator, ConvDecoder, SNNTasNet |
| [`two_speaker_train_v12.py`](two_speaker_train_v12.py) | Stage-2 warmstart fine-tune with PIT loss, EMA, augmentation ablations |
| [`librimix_dataset.py`](librimix_dataset.py) | Libri2Mix dataset loader with spike-safe augmentation |
| [`two_speaker_inference.py`](two_speaker_inference.py) | Inference and SI-SDRi evaluation |
| [`slurm_v12_twospeaker.sh`](slurm_v12_twospeaker.sh) | Self-resubmitting SLURM job for ARC cluster |

### Reference Versions

| File | Version | Notes |
|------|---------|-------|
| [`sep_model_v7.py`](sep_model_v7.py) | v7 | Stateless ChunkedSNN (hidden=256, 3 layers) — best checkpoint |
| [`two_speaker_train_v7.py`](two_speaker_train_v7.py) | v7 | Training with ChunkedSNN separator |
| [`two_speaker_train_v8.py`](two_speaker_train_v8.py) | v8 | Dropout experiment (0.5 — over-regularised) |
| [`two_speaker_train_v9.py`](two_speaker_train_v9.py) | v9 | Gain augmentation experiment (±6 dB — too aggressive) |
| [`two_speaker_train_v10.py`](two_speaker_train_v10.py) | v10 | StatefulSNN attempt (too slow, cancelled) |
| [`two_speaker_train_v11.py`](two_speaker_train_v11.py) | v11 | First completed StatefulGRU run (+5.15 dB) |
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
scp sep_model_v10.py two_speaker_train_v12.py slurm_v12_twospeaker.sh \
    user@arc.csc.ncsu.edu:~/SNNwSoundSeperation/
ssh arc.csc.ncsu.edu
cd ~/SNNwSoundSeperation && sbatch slurm_v12_twospeaker.sh
```

## Inference

```bash
python3 two_speaker_inference.py \
    --model checkpoints_v12a/best_2spk.pt \
    --input mix.wav \
    --out_dir ./pit_output
```
