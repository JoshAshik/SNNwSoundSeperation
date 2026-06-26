# Design Decisions

Answers to "why did we do it this way?"

---

## Architecture

**ConvDecoder replaces single ConvTranspose1d (v7)**
v6 produced a persistent high-frequency hissing artifact. Root cause: binary LIF spikes create
high-freq energy in the masked encoder features. A single ConvTranspose1d upsample has no
mechanism to suppress this — it maps each frequency bin linearly to output samples with no
smoothing. Three Conv1d+GroupNorm+PReLU refinement layers before the upsample act as learnable
low-pass filters applied to the masked features, suppressing spike-pattern noise before reconstruction.
The ConvTranspose1d upsample is kept (and inherits its weights from v6 via v7_starter.pt), so the
model retains the upsampling geometry it already learned. Only the refinement capacity is new.

**v7_starter.pt: encoder-only weight transfer**
The v6 encoder has learned a useful feature representation of speech (despite being trained on FSD50K
mixtures — the low-level time-frequency features transfer across domains). Transferring the encoder
avoids relearning the basic filterbank from scratch. The separator and ConvDecoder refine blocks start
fresh to co-evolve with the Libri2Mix distribution and the new decoder capacity.
`transfer_encoder_decoder()` in sep_model_v7.py handles the v6→v7 copy, including remapping the
ConvTranspose1d weights from v6's decoder into v7's `decoder.upsample`.

**Dataset pivot: FSD50K → Libri2Mix (v7)**
The professor's target is >10 dB SI-SDRi. This benchmark is defined on Libri2Mix clean speech
separation; Conv-TasNet reaches ~14 dB on it. FSD50K is environmental noise (impacts, weather,
alerts) — a fundamentally different domain. The model on FSD50K never saw clean speech pairs, so
it never learned to exploit the harmonic/formant structure that makes speech separable. Additionally,
FSD50K clips are mismatched in duration and content relative to the academic benchmark, so even a
perfect model trained on FSD50K would be measured against a different distribution.
The switch to Libri2Mix aligns training domain, evaluation benchmark, and the pre-trained encoder
weights (originally pre-trained on MiniLibriMix for v3+).

**No PIT (permutation-invariant training)**
8-source PIT = 8! = 40,320 permutations per batch → computationally infeasible.
Channel labels are fixed (ch0=human_voice, etc.) so PIT is unnecessary.
→ Direct per-channel SI-SDR loss with `use_pit=False`.

**snn_chunk=100 frames**
SNN processes 100-frame overlapping chunks in parallel (batched) via overlap-add.
LIF state resets at chunk boundaries. Smaller = less memory; larger = more temporal context.
100 is a balance point for RTX 4060 Ti 16GB.

**EMA decay=0.999**
EMA-smoothed weights give smoother, more stable val metrics than raw weights.
Always use `ema_state["shadow"]` dict for inference and evaluation.

**freeze phase (20 epochs)**
Prevents large gradients from the untrained separator from destroying the pre-trained
encoder weights before the separator has learned basic features.

**Pre-trained encoder (WHAM! 2-source)**
Encoder trained on 800-scene 2-speaker mixtures gives meaningful audio representations.
`transfer_encoder_decoder()` copies only encoder+decoder weights (16,384 params);
leaves separator freshly initialized.

**Matmul fusion in _process_chunk**
The dominant cost was `out_proj = Linear(512, 2048)` called 100× per chunk inside the time loop.
PyTorch Linear/LayerNorm support arbitrary leading dims — applying to `(batch, seg, dim)` is
identical to 100 sequential `(batch, dim)` calls but executes as one large GPU GEMM.
Same fix applied to `in_proj` and `mem_norm`.
The LIF membrane update loop **must remain sequential** (temporal state dependency).

**Stateful separator: LIF state / GRU hidden state persists across chunks (v10/v11)**
v7–v9 used ChunkedSNNSeparator, which resets the LIF membrane state at every chunk boundary
(every 0.1 s). This forces the model to re-identify which speaker is which from scratch in each
0.1 s window — it has no memory of which neurons were active for Speaker A a moment ago.
v10/v11 carry state across all 40 non-overlapping chunks per 4 s utterance. A neuron that started
firing for Speaker A in chunk 1 stays primed for Speaker A in chunks 2–40 (4 s of coherent memory).
This is the primary architectural improvement targeted at the persistent train-val gap.

**Non-overlapping chunks (hop = chunk_size = 100) in v10/v11**
v7 used 50% overlap (hop=50, chunk=100). With state carry, overlapping chunks would process the
same frames at two different membrane states, creating inconsistent gradient signals for the state
transfer. Non-overlapping ensures each encoder frame is processed exactly once per forward pass
and the state handoff is clean. Side effect: 40 outer iterations (vs v7's ~80), which halves the
outer loop overhead.

**StatefulGRUSeparator over StatefulSNNSeparator (v11 over v10)**
StatefulSNNSeparator processes the inner time dimension in a Python for loop (100 iterations per chunk)
because LIF has a sequential state dependency: mem[t] depends on spk[t-1]. This creates 4000 Python-CUDA
dispatches per forward pass on batch=8 — 80× less GPU utilisation than v7's batched design. Observed:
~20 s/batch, ~35,000 s/epoch. Impractical (200 epochs = ~80 days).
StatefulGRUSeparator replaces the inner loop with a single cuDNN GRU kernel: `gru(chunk, h_state)` where
chunk=(B, 100, 256). cuDNN handles the 100 time-step sequential dependency internally in a fused kernel.
Only the outer 40-chunk loop remains in Python. Expected: 2000–4000 s/epoch (~3–4× v7).
The cross-chunk stateful innovation is preserved — GRU hidden state carries between chunks identically
to how LIF membrane state would. The SNN property is lost, but the architectural improvement is kept.

**torch.compile disabled permanently**
`reduce-overhead` mode (CUDA graphs) reserved 977 MB, causing OOM on RTX 4060 Ti 16GB @ batch_size=8.
Even after fixing OOM, compiled backward raised `RuntimeError: inplace operation on version-tracked tensor`.
snntorch's fast-sigmoid surrogate gradient is incompatible with torch.compile's AOT autograd.
Matmul fusion (in _process_chunk) still active and provides the main speedup.

---

## Training

**Stage-2 fine-tune with warmstart and augmentation ablation (v12)**
v11 reached +5.15 dB val SI-SDRi at epoch 200 with LR bottomed at 1e-6. Plain `--resume` would
flatline because cosine schedule hit its floor. Two alternative approaches:
(a) Continue with higher LR — risks destabilising converged weights.
(b) Warmstart: load full model weights (EMA-shadow preferred — the exact weights that achieved
    +5.15 dB), reset optimizer/scheduler/epoch, start fresh with gentle lr=1e-5.
Option (b) is safer because the optimizer has no stale momentum from 200 epochs of training.
The EMA-shadow weights are preferred over raw model_state because val SI-SDRi was measured on
them — they are the weights that actually produced +5.15 dB.

Augmentation disabled for stage-2 because the model is already converged — augmentation helps
generalisation during long training but hurts convergence quality during short fine-tuning of
already-generalised weights. Three ablation runs isolate the effect:
  A (control): dataset augment ON, gain_aug=0 → measures dataset augment contribution
  B (priority): all augment OFF → expected best for converged weights
  C: all augment OFF, lambda_spec=0.75 → tests if stronger spectral loss improves perceptual quality

**v12 ablation conclusions**
Three runs from v11 EMA-shadow weights (lr=1e-5, 100 epochs each):
Run A (dataset augment ON) = +5.46 dB, Run B (all augment OFF) = +5.45 dB, Run C (lambda_spec=0.75) = +5.20 dB.
Key findings: (1) Augmentation is neutral for stage-2 fine-tuning of converged weights — A and B tied.
(2) Increasing lambda_spec from 0.5 to 0.75 degraded val by 0.25 dB — the default spectral weight is better.
(3) Stage-2 warmstart gained +0.31 dB over v11 (+5.15 → +5.46). Modest but confirms the approach works.
(4) The model is near its capacity ceiling — the remaining ~4.5 dB gap to +10 dB requires architectural changes
(e.g. DPRNN/SepFormer separator, wider model, or more training data via train-360).

**DPRNNSeparator replaces StatefulGRUSeparator (v13)**
v12 ablation confirmed the GRU separator hit its capacity ceiling at ~+5.5 dB val SI-SDRi. Train
SI-SDRi was only +6.1 dB — both training and validation were capped, so this is a model capacity
problem, not overfitting. The remaining ~4.5 dB gap to the +10 dB target requires an architectural
change, not more training or hyperparameter tuning.
DPRNN (Dual-Path RNN) replaces the sequential GRU with alternating intra-chunk (fine temporal
resolution) and inter-chunk (full-sequence context) BiLSTMs. The original DPRNN paper achieves ~18 dB
SI-SDRi on WSJ0-2mix. Key design choices:
  - bn_dim=64 bottleneck matches the paper; controls VRAM usage
  - rnn_hidden=128 per direction (256 bidirectional output), projected back to bn_dim
  - 6 blocks (reuses n_layers parameter)
  - chunk_size=200 frames with 50% overlap (hop=100) → ~40 segments for L=3999
  - No Python outer loop — cuDNN handles BiLSTM sequences internally → expected speedup vs v11
  - 2.6M separator params (vs 9.3M GRU) — DPRNN's power is structural, not parameter count
  - lr=5e-4 (fresh separator; conservative start vs paper's 1e-3 since we warmstart encoder)
  - dropout=0.1 (light regularization; train-100 is only 13,900 utterances)
Warmstart: encoder+decoder from v12a EMA shadow (the exact weights that achieved +5.46 dB).
Separator keys filtered out — DPRNNSeparator is freshly initialised. Freeze phase (20 epochs)
protects the transferred encoder from large gradients while the separator learns basic features.

**gain_aug_db = 3.0 not 6.0 (v10/v11)**
v9 applied independent per-source ±6 dB gain augmentation in the training loop. Unknown at the time:
`librimix_dataset.py`'s `spike_safe_augment()` already applies a GLOBAL ±6 dB gain jitter (uniform
across mix+s1+s2 together). The per-source loop augmentation was on top, creating independent ±6 dB
scaling per source → effective ±12 dB SNR swings per utterance. Training collapsed to -1.27 dB
train SI-SDRi. The global jitter is safe (preserves mix=s1+s2); the per-source jitter on top is
what creates the runaway SNR. v10/v11 reduce to ±3 dB per-source (combined max ~±9 dB with dataset jitter).

**CosineAnnealingLR (no warm restarts)**
v1 used CosineAnnealingWarmRestarts — LR resets at ep700 destroyed the best checkpoint.
val > 0 dB occurred in only 17/1,000 epochs, all when LR≈0 near a restart minimum.
→ Replaced with CosineAnnealingLR (monotone cosine over remaining epochs, no resets).

**lambda_rate: 0.005 → 0.03 → 0.10 (v4)**
v1's 0.005 was too weak; spike rates stayed at 0.36.
v2 raised to 0.03 — still insufficient; v3 Layer 2 saturated at 0.52.
v4 raises to 0.10 alongside switching to per-layer L2 (see below).

**Spike rate loss: mean L1 → per-layer L2 with dead zone (v4)**
v3's `(mean_rate - 0.10).abs()` averaged all layers before computing the penalty.
Layer 2 at 0.52 and Layer 3 at 0.09 averaged to 0.30 — a moderate penalty rather than a severe one.
v4 computes L2 per layer with a dead zone [0.10, 0.20]:
  `excess = relu(rate - 0.20) + relu(0.10 - rate);  penalty += excess**2`
A layer at 0.52 now pays (0.32)² = 0.1024 independently of the other layers.
Quadratic growth means saturation at 0.80 pays 4× more than saturation at 0.60.

**lambda_recon: 1.0 → 0.5 → 2.0 (v4) → 5.0 + RMS-MSE (v5)**
v1's 1.0 over-corrected: summed outputs were 6–10× below mixture energy.
v2's 0.5 brought error to ~0.17.
v3 (sigmoid masks) produced reconstruction error of 6.21, ch1 RMS 7× louder than input.
v4 raised to 2.0 with softmax masks — masks sum to 1.0, but decoder weight amplification persisted (error 5.1). Scale-normalised L1 was insufficiently strong.
v5: loss function changed to RMS-normalised MSE (see Decoder weight amplification below) and lambda raised to 5.0. For 7× amplification: MSE contribution = 180 per step vs v4's 12.

**Decoder weight amplification — root cause (v5 fix)**
v4 confirmed masks sum exactly to 1.0 (softmax), so `sum(seps) = decoder(encoder(mix))` by linearity.
Yet reconstruction error remained 5.1 and ch1 RMS was 7× the input.
The cause: the encoder's ReLU discards all negative activations. `decoder(encoder(x)) ≠ x` because
the encoder is non-invertible. The decoder learns to compensate by scaling up its filter weights.
SI-SDR is scale-invariant (gradient w.r.t. output amplitude = 0), so SI-SDR training never resists this.
The v4 recon loss (scale-normalised L1, lambda=2.0) was insufficiently strong to counteract it.
v5 fix: two-part response:
  1. `_recon_loss` → RMS-normalised MSE.  For a 7× amplification: loss = mean((7x-x)²/x²) = 36,
     vs v4's mean-L1/mean_abs ≈ 6.  With lambda_recon=5.0: total contribution = 180 (vs 12 in v4).
  2. `weight_norm` on decoder: decomposes weight = g*v/||v||, separating learned magnitude (g) from
     direction (v). Does NOT hard-constrain amplitude (g is still learned), but improves optimisation
     dynamics and makes the MSE gradient more effective at driving g to smaller values.

**Mask normalization: sigmoid → softmax (v4)**
v3 used independent per-channel sigmoid masks. Two sigmoids can each output up to 1.0, so their product with the encoded features can sum to 2× the original encoded energy. This is what caused the 7× output amplification and reconstruction error of 6.21.
v4 applies `softmax(dim=1)` across the speaker dimension.
For each (freq, time) bin: mask_1 + mask_2 = 1.0 exactly. Energy is partitioned between speakers rather than multiplied.
The network can still assign all energy to one speaker (mask→1.0/0.0) but cannot create energy from nothing.

**RMS-normalised source mixing (v6)**
v1–v5 loaded raw clips and used `scale_to_snr(s1, s2, snr_db)` to balance relative loudness within ±5 dB.
Problem: `scale_to_snr` scales s2 relative to s1's raw RMS — if s1 is intrinsically loud (e.g. a close-mic
voice), the entire mixture is loud; if s1 is quiet (e.g. distant ambient), the mixture is quiet.
The model saw inconsistent absolute energy levels and learned a loudness shortcut: assign the louder source
to one fixed channel. Inference on clips mixed without normalization confirmed this — Speaker 2 SI-SDRi
was -20.91 dB (near-zero output) while Speaker 1 was +8.45 dB (dominated by the louder clip).
v6 fix: RMS-normalise both s1 and s2 to unit energy before `scale_to_snr`. The ±5 dB SNR range then
maps to consistent absolute energies across all scenes and the loudness shortcut is eliminated.
→ `s1 /= s1.rms; s2 /= s2.rms` before mixing in TwoSpeakerMixDataset.__getitem__.

**Channel energy balance loss (v6)**
Even with RMS-normalised mixing, the model could still collapse by outputting near-zero in one channel.
SI-SDR is scale-invariant — a silent channel incurs no gradient penalty from the SI-SDR loss alone.
v6 adds: `balance_loss = (ch_rms[:,0] / ch_rms[:,1]).log().pow(2).mean()` with lambda_balance=0.10.
Log-ratio is zero when both channels carry equal energy; grows symmetrically as one channel dominates.
Works alongside PIT — PIT selects the best permutation assignment; the balance term ensures neither
channel is silenced regardless of which permutation wins.

**lambda_recon: 5.0 → 10.0 (v6)**
v5 achieved recon error 0.39 (synthetic) / 0.75 (real audio). Target is < 0.10.
Doubling lambda_recon to 10.0 doubles the gradient push against amplitude inflation each step.
The MSE contribution for 2× amplification is now 30 (vs 15 in v5), competing more aggressively with
the SI-SDR gradient.

**Spike dead zone widened to [0.10, 0.30] (v6)**
v4/v5 used dead zone [0.10, 0.20]. The model consistently trained at ~0.23–0.25 spike rate, just outside
the upper boundary. This means the penalty was never zero during training — the rate penalty competed
with the SI-SDR loss at every step even when spikes were healthy.
Widening to [0.10, 0.30] puts the natural operating point inside the dead zone, eliminating wasted
gradient. Only true saturation (> 0.30) or dead neurons (< 0.10) incur a penalty.

**LR: 3e-4 → 1.5e-4 and 1e-4 → 5e-5**
Lower LR reduces risk of optimizer overshooting the flat loss landscape of SI-SDR separation.

**scenes_per_epoch=3000 not 10,000**
At 1.62s/batch × 2,500 batches (10k scenes/8 batch) = 67 min/epoch = 47 days for 1,000 epochs.
3,000 scenes → 375 batches → ~7 min/epoch → manageable.

---

## Dataset

**FSD50K music cap=3,500**
Music has ~11,626 training clips vs ~2,000–6,000 for other categories.
Capping at 3,500 prevents the model from over-specialising its music channel.

**FSD50K PESQ disabled**
PESQ is a speech quality metric — meaningless for music/weather/impacts.
Its C library segfaults on non-speech signals.
→ `use_pesq=False` for all FSD50K evaluation calls.

**RAM cache train-only**
Train=4.5 GB, val=0.75 GB, test=2 GB. Caching all three exhausted system RAM and crashed.
Only train is cached: it runs 375+ batches every epoch. Val/test load from disk
(500 batches each, run once per epoch — acceptable with resampler kernel cached).

---

## Checkpoints and resume

**Smart resume (architecture peek)**
`snn_finetune.py` reads `model_cfg` from checkpoint before building the model.
Prevents shape mismatch when DEFAULTS change between runs.

**best_snn_v2_latest.pt saved every epoch**
Enables resume across SLURM jobs. Stores `best_val_si_sdri` separately so the best
checkpoint is never lost even if latest has a worse val score. Resume prefers latest
over best so no epochs are re-run.

**V2 saves to best_snn_v2.pt**
Keeps v1 (best_snn.pt + best_snn_v1_gru_ep696.pt) intact for comparison after v2 completes.

---

## Environment

**python3 not python (local)**
`python` resolves to the dementiaModelOptimized project venv (missing tensorboard).
Always use `python3` for SNNwSoundSeperation.

**No --gres or --mem in SLURM**
`--gres=gpu:1` is invalid on this cluster (partition implies GPU).
`--mem=XG` is not supported on rtx4060ti16g partition.

**sbatch from home dir only**
BeeGFS (`/mnt/beegfs`) is not accessible from the login node.
`sbatch` must be submitted from `~/SNNwSoundSeperation`, not from a BeeGFS path.
