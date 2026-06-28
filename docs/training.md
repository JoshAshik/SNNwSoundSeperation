# Training
<!-- dependencies: two_speaker_train_v17.py (active), two_speaker_train_v15.py (v15/v16 reference), two_speaker_train_v14.py (reference), two_speaker_train_v13.py (reference), snn_finetune.py (8-ch legacy), sep_model_v10.py, oracle_mask_diagnostic.py -->

## Current status (v17 — ACTIVE)

```
Run:          v17 — finer encoder (k=16, s=8) FROM SCRATCH + reference DPRNN recipe
Architecture: sep_model_v10.py  snn_mode=dprnn (DPRNNSeparator, rnn_hidden=128, bn_dim=64)
Script:       two_speaker_train_v17.py
Start ckpt:   NONE — trained end-to-end from scratch (no warmstart, no freeze)
Dataset:      LibriMixDataset — fixed REAL Libri2Mix train-100 pairs (dynamic mixing dropped),
              augmentation OFF (gain_aug=0, train_augment=False)
Key changes:  Single coherent hypothesis — the front-end/loss/schedule were the ceiling, not
              separator capacity or data augmentation (gap≈0 through v13–v16 → underfitting).
                A/D  encoder kernel 32→16, stride 16→8 (~2x frames, ~8000 for a 4s clip).
                     Changing the conv shape makes v13 encoder weights non-transferable, so the
                     whole net trains from scratch with no freeze phase.
                B    pure SI-SDR loss: lambda_recon 5.0→0, lambda_spec 0.5→0.1, lambda_rate=0.
                     Aligns the optimised objective with the reported metric; drops the
                     scale-dependent recon term that dominated the old loss.
                C    fixed real train-100 mixtures (no dynamic mixing — diverged from eval in
                     v15/v16); augmentation off (model underfits, not overfits).
                E    Adam lr=1e-3 + ReduceLROnPlateau (halve, patience 4 val-checks, ABS 0.01 dB
                     threshold), grad clip 5. Replaces cosine-to-1e-6 + 5e-5 enc/dec fine-tune.
              batch_size=8 (stride-8 doubles activations vs v16's stride-16 batch 16).
Gate:         oracle_mask_diagnostic.py — confirm 16:8 STFT-oracle beats 32:16 by >~1 dB on dev
              BEFORE committing the A100 run (bounds the finer front-end's headroom).
Held later:   dprnn_bn_dim 64→128 capacity; train-360 (not yet generated on BeeGFS).
Target:       > +10.0 dB val SI-SDRi (professor's target); bar to beat = v13's +7.22 dB.
Status:       Training on ARC (job 242617, from scratch, self-resubmitting)
```

## Previous status (v16 — COMPLETE)

```
Run:          v16 — dynamic mixing + speed perturbation ONLY (augmentation isolation test)
Architecture: sep_model_v10.py  snn_mode=dprnn (DPRNNSeparator, rnn_hidden=128)
Script:       two_speaker_train_v15.py (reused with flags: --no_train_augment --gain_aug_db 0)
Start ckpt:   checkpoints_v13/best_2spk.pt (encoder+decoder from EMA shadow; separator fresh)
Dataset:      DynamicMixDataset — on-the-fly random speaker pairing, speed perturb 0.95–1.05x,
              relative SNR ±5 dB; spike_safe_augment + per-source gain_aug DISABLED
Result:       best val SI-SDRi = +6.79 dB (ep200), train +8.09 dB, gap +1.3 dB
              Still below v13's +7.22 dB even with augmentation stripped to the minimum —
              confirms dynamic mixing itself diverges from the fixed-mixture eval distribution.
              Motivates v17's return to fixed real mixtures.
Best ckpt:    checkpoints_v16/best_2spk.pt (val=+6.79 dB) — DO NOT OVERWRITE
```

## Previous status (v15 — COMPLETE)

```
Run:          v15 — dynamic clean mixing + speed perturbation (full augmentation stack)
Architecture: sep_model_v10.py  snn_mode=dprnn (DPRNNSeparator, rnn_hidden=128, v13-sized)
Start ckpt:   checkpoints_v13/best_2spk.pt (encoder+decoder from EMA shadow; separator fresh)
Dataset:      DynamicMixDataset — on-the-fly random speaker pairing from s1/+s2/ pool
              speed perturbation 0.95–1.05x, relative SNR ±5 dB, spike_safe_augment ON,
              per-source gain_aug ±3 dB
Key changes:  Dynamic mixing replaces fixed 13,900 Libri2Mix pairs with random combinations;
              v13-sized DPRNN (rnn_hidden=128, bn_dim=64) since v14 wider didn't help;
              combined augmentation stack proved too aggressive
Result:       best val SI-SDRi = +6.64 dB (ep200), train +7.40 dB, gap 1.1 dB
              Below v13's +7.22 dB — augmentation stack over-regularised
Best ckpt:    checkpoints_v15/best_2spk.pt (val=+6.64 dB)
```

## Previous status (v14 — COMPLETE)

```
Run:          v14 — wider DPRNN (rnn_hidden 128→256)
Architecture: sep_model_v10.py  snn_mode=dprnn (DPRNNSeparator, rnn_hidden=256)
Start ckpt:   checkpoints_v13/best_2spk.pt (encoder+decoder from EMA shadow; separator fresh)
Dataset:      Libri2Mix train-100 (13,900 pairs)
Key changes:  rnn_hidden doubled 128→256 (wider BiLSTM); batch_size=16 (A100 headroom);
              warmstart from v13 (not v12a — encoder co-adapted with DPRNN masks);
              separator keys filtered (shape changed); same lr/freeze/augment as v13
Result:       best val SI-SDRi = +7.21 dB (ep200), train +7.83 dB, gap 1.1 dB
              No improvement over v13 — data diversity is the bottleneck, not model width
Best ckpt:    checkpoints_v14/best_2spk.pt (val=+7.21 dB) — DO NOT OVERWRITE
```

## Previous status (v13 — COMPLETE)

```
Run:          v13 — DPRNN separator architecture upgrade
Architecture: sep_model_v10.py  snn_mode=dprnn (DPRNNSeparator, rnn_hidden=128)
Start ckpt:   checkpoints_v12a/best_2spk.pt (encoder+decoder from EMA shadow; separator fresh)
Dataset:      Libri2Mix train-100 (13,900 pairs)
Key changes:  DPRNNSeparator replaces StatefulGRUSeparator; dual-path BiLSTM with 6 blocks;
              bn_dim=64, rnn_hidden=128, chunk_size=200; ~2.6M separator params (vs 9.3M GRU);
              lr=5e-4 (fresh separator), freeze_epochs=20, dropout=0.1;
              gain_aug_db=3.0, train_augment=True
Result:       best val SI-SDRi = +7.22 dB (ep150), train +7.69 dB, gap 0.0–0.6 dB
              +1.76 dB over v12 (+5.46 → +7.22); zero overfitting; LR bottomed at 1e-6
Best ckpt:    checkpoints_v13/best_2spk.pt (val=+7.22 dB) — DO NOT OVERWRITE
```

## Previous status (v12 — COMPLETE)

```
Run:          v12 — Stage-2 fine-tune from v11 EMA-shadow weights
Architecture: sep_model_v10.py  snn_mode=gru_stateful (unchanged from v11)
Start ckpt:   checkpoints_v11/best_2spk.pt (full model warmstart, EMA-shadow preferred)
Dataset:      Libri2Mix train-100 (13,900 pairs)
Key changes:  Warmstart (loads full EMA-shadow weights, resets optimizer/scheduler/epoch);
              lr=1e-5 (vs v11's 1.5e-4); freeze_epochs=0; gain_aug_db=0.0;
              --no_train_augment disables dataset-level spike_safe_augment;
              3-run ablation: A (dataset augment ON), B (all augment OFF), C (lambda_spec=0.75)
Result:       Run A = +5.46 dB (ALL-TIME BEST), Run B = +5.45 dB, Run C = +5.20 dB
              Stage-2 fine-tune gained +0.31 dB over v11 (+5.15 → +5.46)
              Augmentation neutral (A ≈ B); lambda_spec=0.75 hurt (-0.25 dB vs default)
Best ckpt:    checkpoints_v12a/best_2spk.pt (val=+5.46 dB) — DO NOT OVERWRITE
```

### v12 ablation results

| Run | Dataset augment | gain_aug_db | lambda_spec | Val SI-SDRi | Train SI-SDRi | Gap | Verdict |
|-----|----------------|-------------|-------------|-------------|---------------|-----|---------|
| A (control) | ON | 0.0 | 0.5 | **+5.46 dB** | +6.14 dB | 0.7 | **BEST** — augmentation neutral |
| B (priority) | OFF | 0.0 | 0.5 | +5.45 dB | +6.14 dB | 0.7 | Tied with A |
| C | OFF | 0.0 | 0.75 | +5.20 dB | ~+6.2 dB | ~1.0 | Stronger spectral loss hurt |

## Previous status (v11 — COMPLETE)

```
Run:          v11 — StatefulGRUSeparator, Libri2Mix train-100, 200 epochs
Architecture: sep_model_v10.py  snn_mode=gru_stateful
Start ckpt:   checkpoints_v7/best_2spk.pt (encoder+decoder transfer only; separator fresh)
Dataset:      Libri2Mix train-100 (13,900 pairs)
Result:       best val SI-SDRi = +5.15 dB (ep200), train +5.88 dB, gap 0.73 dB
Key changes:  StatefulGRUSeparator — cuDNN GRU processes full (B, 100, 256) chunk in one
              kernel call; hidden state carries between 40 non-overlapping chunks (TBPTT);
              hidden=512, n_layers=6; freeze_epochs=20; gain_aug=±3 dB; lambda_rate=0.0
Checkpoints:  checkpoints_v11/best_2spk.pt (val=+5.15 dB) — DO NOT OVERWRITE
CSV log:      v11_train_log.csv (ARC)
```

## Previous status (v10 — CANCELLED at epoch 0)

```
Run:          v10 — StatefulSNNSeparator — cancelled after ~200 batches of epoch 1
Architecture: sep_model_v10.py  snn_mode=snn_stateful
Result:       CANCELLED — ~20 s/batch → ~35,000 s/epoch (impractical)
Root cause:   4000 Python-CUDA dispatches per forward (40 chunks × 100 LIF time steps × B=8)
              vs v7's 8000 dispatches on effective batch 640 → 80× less GPU utilisation
Resolved by:  v11 StatefulGRUSeparator (single cuDNN kernel per chunk)
```

## Previous status (v9 — COMPLETE)

```
Run:          v9 — Libri2Mix train-100, 200 epochs
Result:       best val SI-SDRi = -2.48 dB (regressed from v7's -1.16 dB)
Final epoch:  train -1.27 dB, gap 1.5 dB (narrowed from v7's 2.9 dB — augmentation working)
Checkpoints:  checkpoints_v9/best_2spk.pt  (val=-2.48 dB)
Problems:     Gain augmentation ±6 dB per-source was DOUBLE-APPLIED: dataset's
              spike_safe_augment already applies global ±6 dB, then training loop added
              independent per-source ±6 dB → effective ±12 dB SNR swings → train collapsed
Resolved by:  v10/v11 gain_aug_db reduced to ±3 dB (safe on top of dataset's global jitter)
```

## Previous status (v8 — COMPLETE)

```
Run:          v8 — Libri2Mix train-100, 200 epochs
Result:       best val SI-SDRi = -1.65 dB (regressed from v7's -1.16 dB)
Final epoch:  train +1.64 dB, gap 3.5 dB (widened from v7's 2.9 dB)
Checkpoints:  checkpoints_v8/best_2spk.pt  (val=-1.65 dB)
Problems:     Dropout 0.5 over-regularised — gap widened and val regressed vs v7
Resolved by:  v9 dropout reduced to 0.35
```

## Previous status (v7 — COMPLETE)

```
Run:          v7 — Libri2Mix train-100, 200 epochs on ARC (~25h wall, multiple SLURM jobs)
Result:       best val SI-SDRi = -1.16 dB  (ep~198)  — ALL-TIME BEST
Final epoch:  ep200: train=+1.75 dB, val=-1.17 dB, gap=2.9 dB, spike=0.126, lr=1.01e-6
Checkpoints:  checkpoints_v7/best_2spk.pt  (val=-1.16 dB)  — DO NOT OVERWRITE
CSV log:      v7_train_log.csv (ARC: ~/SNNwSoundSeperation/)
Problems:     Val never crossed 0 dB — persistent train-val gap (2.9 dB) = overfitting to
              training speaker identities. LIF state reset every 0.1 s forces re-identification
              of speakers from scratch in each chunk.
Resolved by:  v10/v11 stateful separator (cross-chunk state persistence)
```

## Previous status (v6 — COMPLETE)

```
Run:         v6 — FSD50K on-the-fly, 200 epochs on ARC (3 × 6h SLURM jobs)
Result:      val SI-SDRi = +3.82 dB  (FSD50K ceiling)
Checkpoints: checkpoints_v6/best_2spk.pt (ARC)  — DO NOT OVERWRITE
CSV log:     v6_train_log.csv (ARC: ~/SNNwSoundSeperation/)
Problems:    Hissing artifact (high-freq spike-pattern noise leaking through single ConvTranspose1d)
             FSD50K domain mismatch — environmental noise ≠ clean speech separation benchmark
Resolved by: v7 ConvDecoder + Libri2Mix pivot
```

---

## Training history

### v1 (400-scene corpus, 1,202 epochs total)
```
ep 1–20:    freeze phase   val: -0.76 to -1.35 dB
ep 21–696:  finetune       val peaked at +0.04 dB (ep696) → BEST checkpoint
ep 700:     LR warm restart destroyed progress
ep 700–1000: plateau       val: -0.46 to +0.01 dB
```
Key finding: val > 0 in only 17/1000 epochs — cosine warm restarts were the primary blocker.

### v2 (FSD50K, ARC, 100 epochs)
```
ep 1–3:   smoke test (local, freeze phase) — spike rates healthy, masks soft
ep 4–20:  ARC job 204039, freeze phase, val ~0.00 dB (expected)
ep 21+:   finetune phase (encoder unfrozen), train loss dropped to ~0.05 by ep37
ep 100:   COMPLETE — test SI-SDRi +0.05 dB, channel collapse, spike saturation
```

### Comparison table (8-ch track)

| Metric | v1 (ep696) | v2 ep3 (smoke) | v2 ep100 (final) | Target |
|---|---|---|---|---|
| Test SI-SDRi | +0.15 dB | -0.86 dB | +0.05 dB | >+3.0 dB |
| Val SI-SDRi | +0.04 dB | -0.75 dB | ~+0.05 dB | >+2.0 dB |
| Spike rate L1 | 0.36 (sat.) | 0.16 (healthy) | 0.31 (sat.) | ~0.10 |
| Spike rate L2 | 0.18 | 0.19 | 0.60 (sat.) | ~0.10 |
| Spike rate L3 | N/A | 0.19 | 0.21 | ~0.10 |
| Mask saturation | 83.7% | 0% | 68.6% | 10–50% |
| LIF beta | ~1.0001 | clamped 0.5–0.99 | clamped | ~0.9 |

---

### 2-speaker track history

| Version | Dataset | Val SI-SDRi | Key problem |
|---|---|---|---|
| v3 | FSD50K | +3.83 dB | L2 spike sat 0.52, recon error 6.21 (sigmoid masks) |
| v4 | FSD50K | +3.76 dB | Recon error 5.1 (decoder weight amplification) |
| v5 | FSD50K | +3.73 dB | Recon error 0.39, speaker collapse on imbalanced inputs |
| v6 | FSD50K | +3.82 dB | Hissing artifact, FSD50K domain ceiling |
| v7 | Libri2Mix | **-1.16 dB** (ALL-TIME BEST) | train +1.75 dB, gap 2.9 dB; LIF state resets every 0.1 s |
| v8 | Libri2Mix | -1.65 dB | dropout 0.5 over-regularised; gap widened to 3.5 dB |
| v9 | Libri2Mix | -2.48 dB | gain_aug ±6 dB double-applied (dataset + loop = ±12 dB); train collapsed to -1.27 dB |
| v10 | Libri2Mix | CANCELLED | StatefulSNNSeparator: ~20 s/batch → 35,000 s/epoch |
| v11 | Libri2Mix | +5.15 dB | StatefulGRUSeparator; gap 0.73 dB; LR bottomed at 1e-6 |
| v12 | Libri2Mix | +5.46 dB | Stage-2 fine-tune; A=+5.46, B=+5.45, C=+5.20; augmentation neutral |
| v13 | Libri2Mix | **+7.22 dB** (ALL-TIME BEST) | DPRNN separator; +1.76 dB over v12; gap 0.0 dB; capacity ceiling |
| v14 | Libri2Mix | +7.21 dB | Wider DPRNN (rnn_hidden=256); no improvement over v13; data diversity bottleneck |
| v15 | Libri2Mix (dynamic) | +6.64 dB | Dynamic mixing + speed perturb + full augment stack — over-regularised |
| v16 | Libri2Mix (dynamic) | +6.79 dB | Dynamic mixing + speed perturb ONLY — still < v13; dynamic mixing diverges from eval |
| v17 | Libri2Mix (fixed) | ACTIVE | Finer encoder k=16/s=8 from scratch + pure SI-SDR loss + plateau LR; fixed real mixtures |

---

## Hyperparameters (v17 DEFAULTS in two_speaker_train_v17.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 200 | Full training run |
| batch_size | 8 | Stride-8 doubles activations vs v16's batch 16 |
| clip_len | 64,000 | 4s @ 16kHz |
| **kernel_sz** | **16** | **Encoder kernel — was 32 (Experiment A)** |
| **stride** | **8** | **Encoder stride — was 16 → ~8000 frames/4s clip** |
| freeze_epochs | **0** | From scratch — no encoder/decoder to protect |
| warmstart | **NONE** | Trained end-to-end from scratch |
| lr | **1e-3** | Fresh front-end needs a real LR (was 5e-4 + 5e-5) |
| scheduler | **ReduceLROnPlateau** | mode=max, factor=0.5, patience=4 val-checks, ABS 0.01 dB threshold |
| min_lr | 1e-6 | Plateau floor |
| weight_decay | 0.0 | Adam (not AdamW) |
| grad_clip | 5.0 | Global norm clip |
| ema_decay | 0.999 | EMA weights used for val/inference |
| val_every | 5 | Validation at epochs 5,10,15,… |
| max_wall_hours | 5.75 | Exit before ARC 6h limit |
| **lambda_spec** | **0.1** | **Small perceptual term — was 0.5 (Experiment B)** |
| lambda_rate | 0.0 | DPRNN has no spikes |
| **lambda_recon** | **0.0** | **Pure SI-SDR — removed scale-dependent recon (was 5.0)** |
| gain_aug_db | 0.0 | Per-source gain aug off |
| train_augment | False | Dataset spike_safe_augment off (underfitting) |
| dropout | 0.1 | Light regularization |
| dprnn_bn_dim | 64 | Bottleneck (capacity 64→128 held for later) |
| dprnn_rnn_hidden | 128 | v13-sized |
| snn_chunk | 200 | DPRNN internal chunk (~sqrt(2·frames) at stride 8) |

---

## Hyperparameters (v16 — two_speaker_train_v15.py with flags)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 200 | Full training run |
| batch_size | 16 | A100 headroom |
| clip_len | 64,000 | 4s @ 16kHz |
| freeze_epochs | 20 | Separator only for first 20 epochs |
| lr | 5e-4 | Fresh DPRNN separator |
| lr_finetune | 5e-5 | Encoder+decoder LR after unfreeze |
| weight_decay | 1e-4 | AdamW |
| grad_clip | 5.0 | Global norm clip |
| ema_decay | 0.999 | EMA weights used for val/inference |
| val_every | 5 | Validation at epochs 5,10,15,… |
| max_wall_hours | 5.75 | Exit before ARC 6h limit |
| lambda_spec | 0.5 | Multi-resolution spectral L1 |
| lambda_rate | 0.0 | DPRNN has no spikes |
| lambda_recon | 5.0 | Reconstruction MSE |
| gain_aug_db | **0.0** | **DISABLED** — dynamic mixing provides diversity |
| train_augment | **False** | **DISABLED** — spike_safe_augment off |
| speed_perturb | True | 0.95x–1.05x per source |
| epoch_size | 13,900 | Virtual epoch size (matches train-100 pair count) |
| snr_min_db / snr_max_db | -5.0 / +5.0 | Relative SNR between sources |
| warmstart | checkpoints_v13/best_2spk.pt | Encoder+decoder from v13 EMA shadow |
| dropout | 0.1 | Light regularization |
| dprnn_bn_dim | 64 | Bottleneck dimension |
| dprnn_rnn_hidden | 128 | v13-sized (v14 wider didn't help) |
| snn_chunk | 200 | DPRNN internal chunk size |

Result: +6.79 dB — dynamic mixing still below v13.

---

## Hyperparameters (v15 DEFAULTS in two_speaker_train_v15.py)

Same as v16 except: `gain_aug_db=3.0`, `train_augment=True` (both ON).
Result: over-regularised, +6.64 dB < v13's +7.22 dB.

---

## Hyperparameters (v14 DEFAULTS in two_speaker_train_v14.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 200 | Full training run |
| batch_size | 16 | A100 — doubled from v13's 8 |
| clip_len | 64,000 | 4s @ 16kHz |
| dprnn_rnn_hidden | 256 | Per-direction LSTM hidden size (was 128 in v13) |
| gain_aug_db | 3.0 | Per-source gain augmentation |
| train_augment | True | Dataset-level spike_safe_augment enabled |
| warmstart | checkpoints_v13/best_2spk.pt | Encoder+decoder from v13 EMA shadow |
| (other params same as v13) | | |
Result: +7.21 dB — no improvement over v13. Wider rnn_hidden not the bottleneck.

---

## Hyperparameters (v13 DEFAULTS in two_speaker_train_v13.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 200 | Full training run |
| batch_size | 8 | rtx4060ti16g 16 GB |
| dprnn_rnn_hidden | 128 | Per-direction LSTM hidden size |
| warmstart | checkpoints_v12a/best_2spk.pt | Encoder+decoder from EMA shadow |
| (other params same as v14) | | |

---

## Hyperparameters (v12 DEFAULTS in two_speaker_train_v12.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 100 | Shorter than v11 (200) — fine-tuning converged weights |
| batch_size | 8 | rtx4060ti16g 16 GB |
| clip_len | 64,000 | 4s @ 16kHz |
| freeze_epochs | 0 | All parameters trainable from epoch 1 |
| lr | 1e-5 | Gentle fine-tune (v11 ended at 1e-6; this is a 10× reset) |
| weight_decay | 1e-4 | AdamW |
| grad_clip | 5.0 | Global norm clip |
| ema_decay | 0.999 | EMA weights used for val/inference |
| val_every | 5 | Validation at epochs 5,10,15,… |
| max_wall_hours | 5.75 | Exit before ARC 6h limit |
| lambda_spec | 0.5 | Multi-resolution spectral L1 (Run C: 0.75) |
| lambda_rate | 0.0 | GRU has no spikes |
| lambda_recon | 5.0 | Reconstruction MSE |
| gain_aug_db | 0.0 | Per-source gain augmentation disabled |
| train_augment | False | Dataset-level spike_safe_augment disabled (Run A: True) |
| warmstart | checkpoints_v11/best_2spk.pt | Full model load (EMA-shadow preferred) |

---

## Hyperparameters (v7 DEFAULTS in two_speaker_train_v7.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 200 | Self-resubmitting SLURM; max 5.75h wall per job |
| batch_size | 8 | rtx4060ti16g 16 GB |
| clip_len | 64,000 | 4s @ 16kHz |
| freeze_epochs | 20 | Encoder frozen; separator + ConvDecoder refine train freely |
| lr | 1.5e-4 | Full network during freeze phase |
| lr_finetune | 5e-5 | Encoder LR after unfreeze |
| weight_decay | 1e-4 | AdamW |
| grad_clip | 5.0 | Global norm clip |
| ema_decay | 0.999 | EMA weights used for val/inference |
| val_every | 5 | Validation at epochs 5,10,15,… |
| max_wall_hours | 5.75 | Exit before ARC 6h limit |
| lambda_spec | 0.5 | Multi-resolution spectral L1 |
| lambda_rate | 0.10 | Per-layer L2 spike rate penalty (dead zone [0.10, 0.30]) |
| lambda_recon | 5.0 | Reconstruction MSE (conservative vs v6's 10.0 — clean speech has less decoder amplification) |
| target_spike_rate | 0.10 | Lower bound of dead zone |
| target_spike_rate_max | 0.30 | Upper bound of dead zone |

---

## Hyperparameters (v2 DEFAULTS in snn_finetune.py)

| Parameter | Value | Notes |
|---|---|---|
| n_epochs | 100 | Extend to 200–300 for next run |
| batch_size | 8 | 16 OOMs on RTX 4060 Ti 16GB |
| scenes_per_epoch | 3000 | 375 batches/epoch @ ~1.1s/batch |
| freeze_epochs | 20 | Encoder frozen for first 20 epochs |
| lr | 1.5e-4 | Freeze phase (reduced from v1's 3e-4) |
| lr_finetune | 5e-5 | Encoder LR after unfreeze (reduced from v1's 1e-4) |
| scheduler | CosineAnnealingLR | No warm restarts (v2 fix — v1 used WarmRestarts) |
| cosine_T_max | 80 | Remaining epochs after freeze |
| ema_decay | 0.999 | EMA weights used for evaluation |
| val_every | 5 | Validation at epochs 5,10,15,... only |
| max_wall_hours | 5.75 | Exit before ARC 6h wall time |

---

## File inventory

### Active pipeline
| File | Role | Status |
|---|---|---|
| `sep_model_v10.py` | Architecture — DPRNNSeparator + StatefulGRUSeparator + StatefulSNNSeparator + ConvDecoder | Done |
| `two_speaker_train_v17.py` | v17 training — finer encoder (k=16,s=8) from scratch, pure SI-SDR loss, plateau LR, fixed real train-100 | Done |
| `oracle_mask_diagnostic.py` | STFT oracle-mask (IRM/IBM) SI-SDRi ceiling per front-end window — v17 gate (no model/training) | Done |
| `slurm_v17_twospeaker.sh` | ARC SLURM script for v17 (from scratch, batch 8, A100, 6h wall, self-resubmit) | Done |
| `smoke_test_v17.py` | End-to-end smoke test — synthesises tiny Libri2Mix, runs train/resume/oracle to exit 0 | Done |
| `librimix_dataset.py` | Libri2Mix reader — train (v17) + val/test (train_augment param) | Done |
| `eval_dataset.py` | Batch SI-SDRi evaluation on dev/test (--max_samples for dev50) | Done |
| `two_speaker_inference.py` | 2-speaker PIT inference (v11–v17 compatible via sep_model_v10) | Done |

### v15/v16 pipeline (complete — reference)
| File | Role | Status |
|---|---|---|
| `two_speaker_train_v15.py` | v15/v16 training — dynamic mixing + speed perturbation (v13-sized DPRNN) | Complete |
| `dynamic_mix_dataset.py` | On-the-fly clean speech mixing — dropped in v17 (diverged from eval) | Complete |
| `slurm_v16_twospeaker.sh` | ARC SLURM script for v16 (dynamic mix, no extra augment) | Complete |
| `slurm_v15_twospeaker.sh` | ARC SLURM script for v15 (full augment stack) | Complete |

### v14 pipeline (complete)
| File | Role | Status |
|---|---|---|
| `two_speaker_train_v14.py` | v14 training — wider DPRNN (rnn_hidden=256), warmstart from v13 | Complete |
| `slurm_v14_twospeaker.sh` | ARC SLURM script for v14 | Complete |

### v13 pipeline (complete)
| File | Role | Status |
|---|---|---|
| `two_speaker_train_v13.py` | v13 DPRNN (rnn_hidden=128), warmstart from v12a | Complete |
| `slurm_v13_twospeaker.sh` | ARC SLURM script for v13 | Complete |

### v12 pipeline (complete)
| File | Role | Status |
|---|---|---|
| `two_speaker_train_v12.py` | Stage-2 fine-tune — warmstart from v11, lr=1e-5, no augment | Complete |
| `slurm_v12_twospeaker.sh` | ARC SLURM script for v12 (self-resubmitting, 6h wall, 100 epochs) | Complete |

### v11 pipeline (reference)
| File | Role | Status |
|---|---|---|
| `two_speaker_train_v11.py` | v11 training loop (--warmstart added for reuse) | Done |
| `slurm_v11_twospeaker.sh` | v11 ARC SLURM script | Done |

### v10 (reference — architecture only; training script has performance bug)
| File | Role |
|---|---|
| `two_speaker_train_v10.py` | v10 training loop (StatefulSNNSeparator — too slow; bugs fixed but never ran) |
| `slurm_v10_twospeaker.sh` | v10 SLURM script |

### v7 (reference — best checkpoint)
| File | Role |
|---|---|
| `sep_model_v7.py` | v7 architecture (stateless ChunkedSNNSeparator, hidden=256, n_layers=3) |
| `two_speaker_train_v7.py` | v7 training loop |
| `slurm_v7_twospeaker.sh` | v7 SLURM script |

### v3–v6 pipeline (reference — do not modify for v7)
| File | Role |
|---|---|
| `two_speaker_train.py` | v3–v6 training (FSD50K + sep_model.py) |
| `sep_model.py` | v3–v6 architecture (single ConvTranspose1d decoder) |
| `two_speaker_dataset.py` | On-the-fly FSD50K 2-speaker mixer |
| `slurm_v6_twospeaker.sh` | v6 ARC job script |

### 8-channel legacy pipeline (complete — do not modify)
| File | Role |
|---|---|
| `snn_finetune.py` | 8-ch training script |
| `fsd50k_dataset.py` | FSD50K loader + scene synthesizer |
| `sep_inference.py` | 8-ch inference |
| `config.py` | ALL_LABELS, SEP_LABELS constants |
| `wham_pretrain.py` | Stage 1 GRU pre-training |
| `wham_dataset.py` | MiniLibriMix loader |

### Checkpoints
```
LOCAL:
  checkpoints_v6/best_2spk.pt            — v6 best (val=+3.82 dB) — DO NOT OVERWRITE
  checkpoints_v5/best_2spk.pt            — v5 best (val=+3.73 dB)
  checkpoints_v4/best_2spk.pt            — v4 best (val=+3.76 dB)
  checkpoints_snn/best_snn.pt            — v1 best (ep696, val=+0.04 dB) — DO NOT OVERWRITE
  checkpoints_snn/best_snn_v1_gru_ep696.pt — tagged v1 backup — DO NOT OVERWRITE
  checkpoints_pretrain/best_pretrain.pt  — WHAM! encoder/decoder — DO NOT OVERWRITE

ARC (~/SNNwSoundSeperation/):
  checkpoints_v7/best_2spk.pt            — v7 best (val=-1.16 dB) — ALL-TIME BEST — DO NOT OVERWRITE
  checkpoints_v7/best_2spk_latest.pt     — v7 ep200 (training complete)
  checkpoints_v8/best_2spk.pt            — v8 best (val=-1.65 dB)
  checkpoints_v9/best_2spk.pt            — v9 best (val=-2.48 dB)
  checkpoints_v10/                       — empty (v10 cancelled at ~200 batches)
  checkpoints_v11/best_2spk.pt           — v11 best (val=+5.15 dB) — DO NOT OVERWRITE
  checkpoints_v11/best_2spk_latest.pt    — v11 ep200 (training complete)
  checkpoints_v12/best_2spk.pt           — v12 Run B best (val=+5.45 dB)
  checkpoints_v12/best_2spk_latest.pt    — v12 Run B ep100 (training complete)
  checkpoints_v12a/best_2spk.pt          — v12 Run A best (val=+5.46 dB) — DO NOT OVERWRITE
  checkpoints_v12a/best_2spk_latest.pt   — v12 Run A ep100 (training complete)
  checkpoints_v12c/best_2spk.pt          — v12 Run C best (val=+5.20 dB)
  checkpoints_v12c/best_2spk_latest.pt   — v12 Run C ep100 (training complete)
  checkpoints_v13/best_2spk.pt           — v13 best (val=+7.22 dB) — ALL-TIME BEST — DO NOT OVERWRITE
  checkpoints_v13/best_2spk_latest.pt    — v13 ep200 (training complete)
  checkpoints_v14/best_2spk.pt           — v14 best (val=+7.21 dB) — DO NOT OVERWRITE
  checkpoints_v14/best_2spk_latest.pt    — v14 ep200 (training complete)
  checkpoints_v15/best_2spk.pt           — v15 best (val=+6.64 dB)
  checkpoints_v15/best_2spk_latest.pt    — v15 ep200 (training complete)
  checkpoints_v16/best_2spk.pt           — v16 best (val=+6.79 dB) — DO NOT OVERWRITE
  checkpoints_v16/best_2spk_latest.pt    — v16 ep200 (training complete)
  checkpoints_v17/best_2spk.pt           — v17 best (finer encoder from scratch) — IN PROGRESS
  checkpoints_v17/best_2spk_latest.pt    — v17 latest (self-resubmit resume point)
```

### CSV logs
- v1 (8-ch): `snn_finetune_log.csv` — 1,202 rows
- v2 (8-ch): `snn_finetune_v2_log.csv`
- v3–v5 (2-spk): `two_speaker_train_log.csv`, `v4_train_log.csv`, `v5_train_log.csv`
- v6 (2-spk): `v6_train_log.csv` (ARC)
- v7 (2-spk): `v7_train_log.csv` (ARC)
- v8 (2-spk): `v8_train_log.csv` (ARC)
- v9 (2-spk): `v9_train_log.csv` (ARC)
- v11 (2-spk): `v11_train_log.csv` (ARC, complete)
- v12 (2-spk): `v12_train_log.csv` (ARC, complete — Run B, val=+5.45 dB)
- v12a (2-spk): `v12a_train_log.csv` (ARC, complete — Run A, val=+5.46 dB, ALL-TIME BEST)
- v12c (2-spk): `v12c_train_log.csv` (ARC, complete — Run C, val=+5.20 dB)
- v13 (2-spk): `v13_train_log.csv` (ARC, complete — val=+7.22 dB, ALL-TIME BEST)
- v14 (2-spk): `v14_train_log.csv` (ARC, complete — val=+7.21 dB, no improvement over v13)
- v15 (2-spk): `v15_train_log.csv` (ARC, complete — val=+6.64 dB, over-regularised)
- v16 (2-spk): `v16_train_log.csv` (ARC, complete — val=+6.79 dB, dynamic mixing diverges)
- v17 (2-spk): `v17_train_log.csv` (ARC, in progress — finer encoder from scratch)

---

## Commands

### v17: Deploy to ARC and start training (PRIMARY)
```bash
# From Git Bash (Windows)
scp -i "/c/Users/Josh Ashik/private_key.pem" \
  sep_model_v10.py two_speaker_train_v17.py slurm_v17_twospeaker.sh oracle_mask_diagnostic.py \
  smoke_test_v17.py librimix_dataset.py eval_dataset.py two_speaker_inference.py CLAUDE.md \
  juashik@arc.csc.ncsu.edu:~/SNNwSoundSeperation/

# On ARC — GATE FIRST (use the conda python; login-node python3 has no torch)
sed -i 's/\r//' slurm_v17_twospeaker.sh
/mnt/beegfs/juashik/.conda/envs/snn/bin/python oracle_mask_diagnostic.py \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --max_samples 300 --output_csv oracle_dev_v17.csv
# Proceed only if 16:8 oracle beats 32:16 by >~1 dB.

# Then launch training
mkdir -p checkpoints_v17 && sbatch slurm_v17_twospeaker.sh
```

### v17: Evaluate a checkpoint
```bash
python3 eval_dataset.py --model checkpoints_v17/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --output_csv eval_dev_v17.csv

# Fast dev50 for iteration:
python3 eval_dataset.py --model checkpoints_v17/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --max_samples 50
```

### v16: Deploy to ARC and start training (legacy)
```bash
# From Git Bash (Windows)
scp -i "/c/Users/Josh Ashik/private_key.pem" \
  sep_model_v10.py two_speaker_train_v15.py dynamic_mix_dataset.py slurm_v16_twospeaker.sh \
  two_speaker_inference.py eval_dataset.py librimix_dataset.py CLAUDE.md \
  juashik@arc.csc.ncsu.edu:~/SNNwSoundSeperation/

# On ARC (first time only)
sed -i 's/\r//' slurm_v16_twospeaker.sh && mkdir -p checkpoints_v16 && sbatch slurm_v16_twospeaker.sh
```

### v12: Deploy to ARC and start Run B (legacy)
```bash
# From Git Bash (Windows)
scp -i "/c/Users/Josh Ashik/private_key.pem" \
  two_speaker_inference.py two_speaker_train_v11.py two_speaker_train_v12.py \
  eval_dataset.py slurm_v12_twospeaker.sh librimix_dataset.py CLAUDE.md \
  juashik@arc.csc.ncsu.edu:~/SNNwSoundSeperation/

# On ARC (first time only)
sed -i 's/\r//' slurm_v12_twospeaker.sh && mkdir -p checkpoints_v12 && sbatch slurm_v12_twospeaker.sh
```

### v12: Run all 3 ablations in parallel
```bash
# Run B (priority — no augmentation):
sbatch slurm_v12_twospeaker.sh

# Run A (control — dataset augment ON):
mkdir -p checkpoints_v12a
sed 's|CKPT_DIR=./checkpoints_v12|CKPT_DIR=./checkpoints_v12a|; s|--no_train_augment|--train_augment --log_dir ./runs_v12a --csv_path ./v12a_train_log.csv|' slurm_v12_twospeaker.sh > slurm_v12a.sh
sed -i 's/snn_v12/snn_v12a/g' slurm_v12a.sh
sbatch slurm_v12a.sh

# Run C (lambda_spec=0.75):
mkdir -p checkpoints_v12c
sed 's|CKPT_DIR=./checkpoints_v12|CKPT_DIR=./checkpoints_v12c|; s|--no_train_augment|--no_train_augment --lambda_spec 0.75 --log_dir ./runs_v12c --csv_path ./v12c_train_log.csv|' slurm_v12_twospeaker.sh > slurm_v12c.sh
sed -i 's/snn_v12/snn_v12c/g' slurm_v12c.sh
sbatch slurm_v12c.sh

# Verify all three queued:
squeue -u juashik
```

### v12: Evaluate a checkpoint
```bash
python3 eval_dataset.py --model checkpoints_v12/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --output_csv eval_dev_v12.csv

# Fast dev50 for iteration:
python3 eval_dataset.py --model checkpoints_v12/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --max_samples 50
```

### v11: Requeue after interruption (legacy)
```bash
cd ~/SNNwSoundSeperation && sbatch slurm_v11_twospeaker.sh
```

### v7: Requeue (reference — if ever needed)
```bash
cd ~/SNNwSoundSeperation && sbatch slurm_v7_twospeaker.sh
```

### ARC: Monitor
```bash
squeue -u juashik
tail -f ~/SNNwSoundSeperation/slurm_JOBID.out
cat ~/SNNwSoundSeperation/slurm_JOBID.err
```

### ARC: Interactive session
```bash
salloc -n 1 -p rtx4060ti16g
exit   # release when done
```

### v7: Smoke test (once pipeline files exist)
```bash
python3 two_speaker_train_v7.py --n_epochs 2 --batch_size 4 --num_workers 0 \
    --librimix_root ./data/librimix   # local path, if data available
```

### Libri2Mix data check (ARC compute node)
```bash
salloc -n 1 -p rtx4060ti16g
ls /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max/train-100/mix_clean/ | wc -l   # expect 13900
ls /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max/dev/mix_clean/ | wc -l         # expect 3000
```

### 8-ch legacy inference
```bash
python3 sep_inference.py --checkpoint checkpoints_snn/best_snn_v2.pt --from_dataset --data_root ./data/fsd50k --n_samples 10
```
