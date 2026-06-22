# Dataset
<!-- dependencies: librimix_dataset.py (active), fsd50k_dataset.py (v3–v6), sep_dataset.py, config.py -->

---

## Libri2Mix (v7 — ACTIVE)

Pre-generated 2-speaker clean speech mixtures from LibriSpeech. Academic standard benchmark for blind source separation.

### Data paths (ARC BeeGFS)
```
/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max/
  train-100/
    mix_clean/   13,900 WAV files  — mono 16kHz, variable length (max-mode: padded to longer source)
    s1/          13,900 WAV files  — speaker 1 source
    s2/          13,900 WAV files  — speaker 2 source
  dev/
    mix_clean/   3,000 WAV files
    s1/          3,000 WAV files
    s2/          3,000 WAV files
  test/
    mix_clean/   3,000 WAV files
    s1/          3,000 WAV files
    s2/          3,000 WAV files

/mnt/beegfs/juashik/librimix/LibriSpeech/   — source audio (train-clean-100/360, dev-clean, test-clean)
/mnt/beegfs/juashik/librimix/wham_noise/    — WHAM noise (tr/ files unreadable — train-360 not generated)
```

**train-360 not available:** WHAM noise `tr/` files on BeeGFS return `System error` when read. train-100 (13,900 mixtures) is sufficient — Conv-TasNet achieves ~14 dB on it.

### File naming
Each file is named `<utt1_id>_<utt2_id>.wav`. The `s1/` and `s2/` files share the same stem as `mix_clean/`. The dataset loader pairs them by stem.

### "max" mode
Sources are zero-padded to the length of the longer utterance (not truncated). Mixtures are variable length. The dataset loader handles this by cropping/padding to a fixed training length (e.g. 4s = 64,000 samples at 16kHz).

### Dataset handler (`librimix_dataset.py`)
Reads pre-generated wav files — no on-the-fly mixing. Returns `(mixture, s1, s2)` tensors at fixed
`clip_len=64000` (4s). Crops or tile-pads all three wavs with the same start offset so `mix = s1 + s2`
is preserved after cropping. Peak-normalises using the mixture peak (uniform scale applied to all three).
Applies spike-safe augmentation (gain jitter ±6 dB, polarity inversion, circular shift, additive noise,
cosine taper) during training when `augment=True`.

`build_librimix_dataloaders()` accepts a `train_augment` parameter (default: `True`). Setting
`train_augment=False` disables all dataset-level augmentation for the training split. This is used
by v12 stage-2 fine-tuning (`--no_train_augment`) to avoid perturbing converged weights. Note that
this is separate from the per-source gain augmentation in the training script (`gain_aug_db`) — both
can be independently controlled.

Smoke test: `python3 librimix_dataset.py <librimix_root>`.

### Why Libri2Mix instead of FSD50K
FSD50K is environmental noise (impacts, weather, alerts). The academic 10 dB SI-SDRi target is defined on clean speech separation (Libri2Mix). Domain mismatch was the primary ceiling for v3–v6. See decisions.md for full rationale.

---

## Output channel assignment (8-ch legacy — fixed, no permutation)

```
ch0 → human_voice
ch1 → animal
ch2 → impacts
ch3 → music
ch4 → weather
ch5 → alerts
ch6 → domestic_and_objects
ch7 → tools_and_machines
```
Defined in `config.py:ALL_LABELS`. Imported as `SEP_LABELS` in `sep_dataset.py` and `fsd50k_dataset.py`.

---

## FSD50K (v3–v6 — legacy 2-speaker track)

### Paths
```
ARC BeeGFS:  /mnt/beegfs/juashik/fsd50k/
  FSD50K.dev_audio/    — 40,966 WAVs @ 44.1kHz
  FSD50K.eval_audio/   — 10,231 WAVs @ 44.1kHz
  FSD50K.ground_truth/dev.csv    (fname, labels, mids, split)
  FSD50K.ground_truth/eval.csv
  FSD50K.ground_truth/vocabulary.csv

Local:  data/fsd50k/   (zip archives deleted after extraction — ~35 GB freed)
```

### Clips per label (training split, after mapping)
```
human_voice:          6,041
domestic_and_objects: 5,174
tools_and_machines:   4,195
music:                3,500  (capped — raw ~11,626)
impacts:              3,309
animal:               2,898
weather:              2,574
alerts:               1,784
Total train clips:    29,475
```

Music is capped at 3,500 to prevent over-specialisation of the music channel.

### Scene generation
- `FSD50KDataset.__getitem__`: picks 2–4 categories randomly, one clip per category
- Clips summed with random gain (±6 dB), peak-normalised to 3s / 16kHz mono
- `scenes_per_epoch=3000` → 375 batches/epoch (at batch_size=8)
- Val/test: pre-generated fixed scenes (2,000 each), seeded for reproducibility

### RAM cache (train only)
- All 29,475 train clips pre-loaded to RAM at job start (~4.5 GB, ~20–25 min on BeeGFS)
- Val/test load from disk (cache_audio=False) — runs once per epoch, acceptable
- Only train is cached: caching all three splits exhausted RAM and crashed
- Resampler kernel (44.1→16kHz sinc) cached module-level in `_RESAMPLER_CACHE`

### FSD50K → 8-label mapping
Full dict `FSD50K_TO_LABEL` in `fsd50k_dataset.py` (~80 entries, lines ~40–260).
Key: use underscores (`"Male_speech_and_man_speaking"` not `"Male speech, man speaking"`).

### Sanity check
```bash
python3 fsd50k_dataset.py data/fsd50k
```

---

## Original 400-scene corpus (v1 — for comparison/inference only)

```
Root:   ./data/
Map:    ./original_data_map.json
Clips:  data/sources/, data/transients/, data/shared/
Labels: labels.csv  (400 rows: id, source_labels, transient_labels, dynamic_labels)
Split:  seed=42 → 75% train (~300), 15% val (~60), 10% test (~40)
Format: 3s, 16kHz, mono
```

Clip counts per label (imbalanced):
```
domestic=240, tools=240, alerts=200, animal=200,
human=200, impacts=200, weather=120, music=80
```

---

## Augmentation pipeline (training only)

1. **Cross-scene remix** (50%): borrow transient clips from a different scene
2. **Gain jitter** (100%): ±6 dB random scaling
3. **Noise injection** (100%): 0–2% white noise added to mixture only
4. **Circular time shift** (50%): roll up to 0.5s
5. **SpecAugment** (60%): zero a random freq band + time window in mixture STFT
6. **Repeat factor** (4×): each scene appears 4× per epoch with independent random augmentation

Val and test sets: **no augmentation ever**.

All augmentation lives in `sep_dataset.py:SoundSepDataset.__getitem__` (original corpus)
and `fsd50k_dataset.py:FSD50KDataset.__getitem__` (FSD50K).

---

## WHAM! pre-training data (Stage 1 — complete)

```
Path:   wham_data/MiniLibriMix/   (local)
        train/ (800 scenes) + val/ (200 scenes)
Format: 2-speaker clean speech mixtures, 8kHz resampled to 16kHz
Use:    Stage 1 GRU pre-training only — not needed once best_pretrain.pt exists
```

---

## Two-Speaker Mixture Dataset (TwoSpeakerMixDataset)

Defined in `two_speaker_dataset.py`. Designed for PIT pre-training (`n_speakers=2`)
before or alongside 8-channel fine-tuning. Returns `(mixture, target_1, target_2)` —
all `(L,)` tensors, peak-normalised.

### Constructor parameters
| Parameter | Default | Notes |
|---|---|---|
| `clip_index` | required | `[(wav_path, label_str), ...]` — any flat clip list |
| `scenes_per_epoch` | 5000 | Synthetic epoch size |
| `overlap_range` | (0.3, 1.0) | Fraction of s2 that overlaps s1; 1.0=fully simultaneous |
| `snr_range` | (-5.0, 5.0) | dB of s1 relative to s2; positive=s1 louder |
| `augment` | True | Apply spike-safe augmentations (training only) |

### Mixing pipeline (per `__getitem__`)
1. Load two distinct clips via `load_and_fit` (random-crop or tile-pad to 3s)
2. Scale s2 to target SNR relative to s1 via RMS ratio
3. Place s2 at a random start offset within `overlap_range` (jittered each call)
4. Sum → peak-normalise mixture; apply same scale to both targets
5. Apply `spike_safe_augment` if `augment=True`

### Overlap ratio semantics
```
overlap_ratio=1.0  →  s2 starts at sample 0  (fully simultaneous)
overlap_ratio=0.5  →  s2 may start up to 50% of L into s1
overlap_ratio=0.0  →  s2 starts at or after s1 ends (truncated to L)
```
Non-overlapping regions contain only one source — model must learn to silence the absent channel.

### PIT collation (in training loop)
```python
mixture, t1, t2 = batch                   # (B, L) each
targets = torch.stack([t1, t2], dim=1)    # (B, 2, L)
# Try both permutations, keep assignment with lower SI-SDR loss
```

---

## Spike-safe augmentation rules

Applied uniformly to mixture AND all targets to preserve PIT alignment.

| Augmentation | Applied to | Safe? | Reason |
|---|---|---|---|
| Gain jitter ±6 dB | mix + targets | ✅ | Uniform scale — no discontinuity |
| Polarity inversion | mix + targets | ✅ | Conv1d encoder learns both polarities |
| Circular time shift (≤0.75s) | mix + targets | ✅ | `torch.roll` — no edge artefact |
| Low-level additive noise (<1.5% RMS) | mix only | ✅ | Targets stay clean for loss |
| Smooth cosine amplitude taper | mix + targets | ✅ | No abrupt step edge |
| Hard rectangular zero-mask | — | ❌ | Step edge → encoder transient → spurious LIF burst |
| Waveform SpecAugment (ISTFT) | — | ❌ | Gibbs ringing at band edges mimics fake events |
| Phase scrambling / large pitch shift | — | ❌ | Destroys event timing that LIF integrates |
