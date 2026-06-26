# Architecture
<!-- dependencies: sep_model_v10.py (active), sep_model_v7.py (v7 reference), sep_model.py (v3–v6 reference) -->

## SNNTasNet model graph (v13 — ACTIVE, snn_mode="dprnn")

```
Input: (B, T)  — mono mixture, T=64000 (4s @ 16kHz)

Encoder:   Conv1d(1→256, kernel=32, stride=16, ReLU)         [transferred from v12a best]
           → (B, 256, L)  where L=3999 (ceil((T-k)/s)+1)
           SpecAugment (training only): zero 2 random freq bands

Separator: DPRNNSeparator  [snn_mode="dprnn"]                [NEW in v13]
  1. GroupNorm(1, 256) + Conv1d(256→64, k=1)  — bottleneck to bn_dim=64
  2. Pad L→L_pad, unfold(K=200, H=100)        — ~40 segments with 50% overlap
  3. 6× DPRNNBlock:
       Intra-chunk: (B*S, K, 64) → BiLSTM(64, 128) → Linear(256→64) → residual+GroupNorm
       Inter-chunk: (B*K, S, 64) → BiLSTM(64, 128) → Linear(256→64) → residual+GroupNorm
  4. Overlap-add reconstruct → (B, 64, L)
  5. Conv1d(64→512, k=1) → view(B, 2, 256, L) → softmax(dim=1) → masks
  spike_recs = []  (no LIF neurons)

Decoder:   ConvDecoder  [transferred from v12a best — unchanged since v7]
  - 3 × Conv1d(256,256,k=3)+GroupNorm(8,256)+PReLU
  - ConvTranspose1d(256→1, k=32, s=16)
  masked encoder features → refine → upsample → (B*2, 1, T') → reshape → (B, 2, T)

Output: {"separated": (B, 2, T), "masks": (B, 2, 256, L), "spike_recs": []}
```

**Params:** ~3.2M total (2.6M separator, 0.6M encoder+decoder).
**Speed:** No Python loop — full sequence processed at once via cuDNN BiLSTM. Expected ~1500–3000 s/epoch.

---

## SNNTasNet model graph (v11 — reference, snn_mode="gru_stateful")

```
Input: (B, T)  — mono mixture, T=64000 (4s @ 16kHz)

Encoder:   Conv1d(1→256, kernel=32, stride=16, ReLU)         [transferred from v7 best]
           → (B, 256, L)  where L=4000 (T/stride)
           SpecAugment (training only): zero 2 random freq bands

Separator: StatefulGRUSeparator  [snn_mode="gru_stateful"]   [NEW in v11]
  - LayerNorm(256) applied to encoder output
  - Split into 40 non-overlapping chunks of 100 frames (hop=chunk=100)
  - For each chunk (40 Python iterations):
      GRU(input=256, hidden=512, n_layers=6, bidirectional=False)
      processes full (B, 100, 256) chunk in a single cuDNN kernel call
      → (B, 100, 512)
      Linear(512 → 256*2)  → (B, 100, 512)
      h_state detached at chunk boundary (TBPTT)
  - Reconstruct full sequence → (B, 512, L)
  - view(B, 2, 256, L) → softmax(dim=1) → masks
  - spike_recs = []  (no LIF neurons)

Decoder:   ConvDecoder  [transferred from v7 best — unchanged since v7]
  - 3 × Conv1d(256,256,k=3)+GroupNorm(8,256)+PReLU
  - ConvTranspose1d(256→1, k=32, s=16)
  masked encoder features → refine → upsample → (B*2, 1, T') → reshape → (B, 2, T)

Output: {"separated": (B, 2, T), "masks": (B, 2, 256, L), "spike_recs": []}
```

**Speed:** 40 Python loop iterations per forward pass (one cuDNN GRU call each).
Expected ~2000–4000 s/epoch. Compare: v10 LIF had 4000 Python-CUDA dispatches → ~35,000 s/epoch.
Superseded by v13 DPRNNSeparator (no Python loop, all cuDNN).

---

## SNNTasNet model graph (v10 — reference, snn_mode="snn_stateful")

```
Separator: StatefulSNNSeparator  [v10 design, too slow to train]
  - 40 non-overlapping chunks of 100 frames (same segmentation as v11)
  - For each chunk (40 Python iterations):
      For each time step (100 Python iterations):
          n_layers × LIFResidualBlock(hidden=512)
          → 4000 Python-CUDA dispatches per forward pass @ B=8
  - LIF membrane state carries between chunks (same cross-chunk memory as v11)
  - ~20 s/batch observed → ~35,000 s/epoch → impractical
```

hidden=512, n_layers=6. Defined in sep_model_v10.py. Kept for reference; never completed training.

---

## SNNTasNet model graph (v7 — reference, snn_mode="snn")

```
Input: (B, T)  — mono mixture, T=64000 (4s @ 16kHz)

Encoder:   Conv1d(1→256, kernel=32, stride=16, ReLU)
           SpecAugment (training only)

Separator: ChunkedSNNSeparator  [snn_mode="snn"]
  - Overlapping chunks (chunk_size=100, hop=50) → ~80 chunks per 4s clip
  - ALL chunks batched together: effective batch = B*n_segs = 640
  - in_proj:  LayerNorm(256) → Linear(256→hidden)
  - n_layers × LIFResidualBlock:
      fc: Linear(hidden→hidden), bn: BatchNorm1d(hidden)
      lif: snn.Leaky(beta clamped [0.5, 0.99], threshold learnable)
      residual: output = lif(fc(bn(x))) + x
  - mem_norm: LayerNorm(hidden), out_proj: Linear(hidden*2 → 256*2)
  - LIF state RESETS at each chunk boundary (stateless — v11 key improvement)
  - Overlap-add reconstruction → softmax masks

Decoder:   ConvDecoder (same as v11)
```

hidden=256, n_layers=3. Defined in sep_model_v7.py. Best checkpoint: -1.16 dB (all-time best).

**GRU mode** (`snn_mode="gru"`): ChunkedGRUSeparator — bidirectional GRU, ALL chunks batched in parallel. Used for reference/pre-training only.
**Stateful GRU** (`snn_mode="gru_stateful"`): StatefulGRUSeparator — v11 default, sequential chunks with carried state. In sep_model_v10.py.

---

## SNNTasNet model graph (v3–v6 — reference)

```
Decoder:   ConvTranspose1d(256→1, kernel=32, stride=16)  [single layer, no refinement]
           mask[c] * encoded → decoder → (B, T)
```

v3–v6 implemented in `sep_model.py`. v7 implemented in `sep_model_v7.py`.

---

## Configs

### v1 checkpoint (locked — do not modify):
```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":192, "n_layers":2, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":8}
# params: 930,182
```

### v2 DEFAULTS (snn_finetune.py):
```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":256, "n_layers":3, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":8}
# params: ~1,331,718
```

---

## LIF neuron dynamics

```
U[t] = β·U[t-1] + I[t]          # integrate
S[t] = Θ(U[t] - θ)              # fire (Heaviside step)
U[t] ← U[t]·(1 - S[t])         # reset after spike
```
- β (decay): learnable, **clamped [0.5, 0.99]** — prevents saturation to β≈1.0 (v2 fix)
- θ (threshold): learnable
- Surrogate gradient: fast-sigmoid (slope=25) via snntorch — enables backprop through spikes
- Target spike rate: ~0.10 (10%)

---

## Loss function (v2 values)

```python
SeparationLoss(
    lambda_spec    = 0.5,   # multi-resolution spectral (L1 on log-STFT @ FFT 256/512/1024)
    lambda_silence = 0.3,   # penalise energy in silent channels
    lambda_rate    = 0.03,  # spike rate regularisation (target 0.10) — raised from v1's 0.005
    lambda_recon   = 0.5,   # reconstruction: F.l1_loss(estimates.sum(1), mixture) — halved from v1's 1.0
    use_pit        = False, # fixed 8-channel assignment — no permutation search
)
# total = SI_SDR_loss + lambda_spec*spec + lambda_silence*sil + lambda_rate*rate + lambda_recon*recon
# forward: losses = loss_fn(estimates, targets, spike_recs, mixture=mixture)
# returns dict: total, si_sdr_loss, spec_loss, silence_loss, recon_loss, rate_loss
```

---

## Checkpoint format

```python
{
    "epoch":           int,
    "model_state":     state_dict,        # raw weights
    "ema_state":       {"shadow": ...,    # USE THESE for inference
                        "decay": 0.999},
    "optimizer_state": state_dict,
    "scheduler_state": state_dict,
    "val_si_sdri":     float,
    "model_cfg":       dict,              # architecture dict — read by sep_inference.py
}
# Load with: torch.load(path, map_location='cpu', weights_only=False)
# sep_inference.py reads model_cfg automatically — no need to specify arch manually
```

---

## v4 config (two_speaker_train.py DEFAULTS)

```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":256, "n_layers":3, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":2,
 "lambda_rate":0.10, "lambda_recon":2.0,
 "target_spike_rate":0.10, "target_spike_rate_max":0.20,
 "use_weight_norm":False}
# checkpoints: checkpoints_v4/best_2spk.pt  |  val SI-SDRi: +3.76 dB
```

---

## v5 config (two_speaker_train.py DEFAULTS)

```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":256, "n_layers":3, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":2,
 "lambda_rate":0.10, "lambda_recon":5.0,
 "target_spike_rate":0.10, "target_spike_rate_max":0.20,
 "use_weight_norm":True}
# checkpoints: checkpoints_v5/best_2spk.pt
```

---

## v6 config (two_speaker_train.py DEFAULTS)

```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":256, "n_layers":3, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":2,
 "lambda_rate":0.10, "lambda_recon":10.0, "lambda_balance":0.10,
 "target_spike_rate":0.10, "target_spike_rate_max":0.30,
 "use_weight_norm":True}
# checkpoints: checkpoints_v6/best_2spk.pt (ARC)  |  val SI-SDRi: +3.82 dB
# dataset: FSD50K on-the-fly mixing
```

---

## v7 config (two_speaker_train_v7.py)

```python
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":256, "n_layers":3, "dropout":0.3,
 "snn_mode":"snn", "snn_chunk":100, "n_speakers":2,
 "decoder_refine":3, "decoder_groups":8}
# clip_len: 64,000 samples (4s @ 16kHz)
# starting checkpoint: v7_starter.pt (encoder from v6, separator+ConvDecoder fresh)
# dataset: Libri2Mix train-100 (13,900 pre-generated pairs on ARC BeeGFS)
# result: val=-1.16 dB (all-time best)
```

---

## v13 config (sep_model_v10.py)

```python
# v13 (dprnn — ACTIVE):
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":128, "n_layers":6, "dropout":0.1,
 "snn_mode":"dprnn", "snn_chunk":200, "n_speakers":2,
 "decoder_refine":3, "decoder_groups":8, "use_weight_norm":False,
 "dprnn_bn_dim":64, "dprnn_rnn_hidden":128}
# starter: checkpoints_v12a/best_2spk.pt (encoder+decoder only; separator fresh)
# params: ~3,241,536 total, ~2,633,024 separator
```

---

## v10/v11 config (sep_model_v10.py)

```python
# v11 (gru_stateful — ACTIVE):
{"n_filters":256, "kernel_sz":32, "stride":16,
 "hidden":512, "n_layers":6, "dropout":0.35,
 "snn_mode":"gru_stateful", "snn_chunk":100, "n_speakers":2,
 "decoder_refine":3, "decoder_groups":8, "use_weight_norm":False}
# starter: checkpoints_v7/best_2spk.pt (encoder+decoder only; separator fresh)
# lambda_rate=0.0 (no spikes), gain_aug_db=3.0, freeze_epochs=20

# v10 (snn_stateful — reference, never completed):
# {"snn_mode":"snn_stateful", ...}  — same dims as v11 but LIF separator
```

---

## v4/v5 Loss function

```python
total = (
    si_sdr_loss                                   # PIT SI-SDR (primary)
    + 0.5  * spec_loss                            # multi-res spectral L1 (unchanged)
    + 0.10 * rate_loss                            # per-layer L2 spike penalty
    + λ    * recon_loss                           # energy conservation (v4: 2.0, v5: 5.0)
)
```

**Spike rate loss (v4+):** Per-layer L2 with dead zone [0.10, 0.20]:
```python
for each LIF layer:
    excess = relu(rate - 0.20) + relu(0.10 - rate)
    penalty += excess ** 2     # quadratic, zero inside band
```
A layer at 0.52 pays (0.52−0.20)² = 0.1024 per forward pass.
v3's mean-L1 let Layer 2 (0.52) hide behind Layers 1 and 3 being near-target.

**Mask normalization (v4+):** `softmax(dim=1)` replaces `sigmoid`.
Masks sum to exactly 1.0 across the speaker dimension for every (freq, time) bin.
`sum(seps) = decoder(encoder(mix))` exactly — energy partitioned, not multiplied.

**Reconstruction loss (v5):** RMS-normalised MSE replaces scale-normalised L1.
```python
mix_power = (mixture**2).mean(dim=-1, keepdim=True)   # RMS²
recon_loss = ((summed - mixture)**2 / mix_power).mean()
```
For a 7× amplification: loss = mean((7x−x)²/x²) = 36.
With lambda_recon=5.0: contribution = 180  (vs v4: ~12). Competes with SI-SDR gradient.

**Decoder weight normalisation (v5):** `weight_norm` applied to ConvTranspose1d decoder.
Decomposes weight = g·v/‖v‖ (magnitude g, direction v). Does not hard-constrain g — the
MSE loss provides the actual amplitude constraint. Weight_norm improves optimisation dynamics
so the gradient drives g to smaller values more reliably. Stored in `model_cfg["use_weight_norm"]`
so inference scripts rebuild the model correctly.

---

## Key code locations

| What | File | Notes |
|---|---|---|
| DPRNNBlock | `sep_model_v10.py` | v13 dual-path block (intra + inter chunk BiLSTM) |
| DPRNNSeparator | `sep_model_v10.py` | v13 active separator |
| StatefulGRUSeparator | `sep_model_v10.py` | v11/v12 separator (reference) |
| StatefulSNNSeparator | `sep_model_v10.py` | v10 design (too slow) |
| LIFResidualBlock (beta clamp) | `sep_model_v10.py` | shared with v7 |
| ChunkedSNNSeparator | `sep_model_v10.py` | v7 stateless LIF (kept for reference) |
| ConvDecoder | `sep_model_v10.py` | unchanged from v7 |
| SNNTasNet constructor | `sep_model_v10.py` | snn_mode dispatch |
| transfer_encoder_decoder | `sep_model_v10.py` | v7→v11 encoder+decoder copy |
| PIT forward pass + loss | `two_speaker_train_v11.py` | pit_forward() |
| ChunkedSNNSeparator (v7 original) | `sep_model_v7.py` | reference only |
| SeparationLoss (8-ch legacy) | `sep_losses.py` | v3–v6 only |
