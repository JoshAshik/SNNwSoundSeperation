# Bugs & Fixes

## Fixed — v1 session

| # | Symptom | Fix | File |
|---|---|---|---|
| 1 | PIT permutation explosion (8!=40,320) | Direct O(C) assignment — no PIT for 8 channels | `sep_metrics.py` |
| 2 | `UnicodeEncodeError` on print | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at top | all scripts |
| 3 | TF/oneDNN warnings on import | `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` before imports | all scripts |
| 4 | Spike rate measured residual+x, not true binary spike | Fixed in forward() to return spike tensor only | `sep_model.py` |
| 5 | Resume shape mismatch when DEFAULTS change | Read architecture from checkpoint BEFORE model build | `snn_finetune.py:~278` |
| 6 | CSV header missing `recon_loss` column | Auto-corrects on resume | `snn_finetune.py` |
| 7 | Inference active-channel detection always-on | `separate()` returns raw wav; caller decides silence | `sep_inference.py` |
| 8 | snntorch missing from `python` env | Installed snntorch 0.9.4 in both envs | env |

## Fixed — v2 session (correctness)

| # | Symptom | Fix | File |
|---|---|---|---|
| 9 | LIF beta drifts to ≈1.0 → layer saturation | `self.lif.beta.data.clamp_(0.5, 0.99)` in forward() | `sep_model.py:~104` |
| 10 | `lambda_recon=1.0` over-corrected (6–10× energy underestimation) | Changed default to 0.5 | `sep_losses.py` |
| 11 | `lambda_rate=0.005` too weak → spike rates stay high | Changed default to 0.03 | `sep_losses.py` |
| 12 | Cosine warm restarts reset LR and destroyed progress at ep700 | Replaced with `CosineAnnealingLR` (no restarts) | `snn_finetune.py` |
| 13 | Max LR too high | lr: 3e-4→1.5e-4, lr_finetune: 1e-4→5e-5 | `snn_finetune.py` |
| 14 | PESQ segfaults on non-speech audio | Added `use_pesq=True` param; FSD50K passes `use_pesq=False` | `sep_metrics.py:~54` |

## Fixed — ARC migration (2026-03-26)

| # | Symptom | Fix | File |
|---|---|---|---|
| 15 | `torch.compile` before resume caused checkpoint key mismatch | Moved compile to after resume block | `snn_finetune.py` |
| 16 | `torch.compile reduce-overhead` OOM: CUDA graphs reserved 977 MB | `torch.compile` removed entirely | `snn_finetune.py` |
| 17 | Compiled backward: `RuntimeError: inplace op on version-tracked tensor` | snntorch surrogate gradient incompatible with AOT autograd — compile disabled permanently | `snn_finetune.py` |

## Fixed — ARC 6h wall-time (2026-03-26)

| # | Symptom | Fix | File |
|---|---|---|---|
| 18 | 1000 epochs × 447s = 124h >> 6h limit | Lowered to 100 epochs; added `--max_wall_hours 5.75` + `--val_every 5` | `snn_finetune.py`, `run_train.slurm` |
| 19 | No resume across jobs — re-ran from ep1 | Added `best_snn_v2_latest.pt` (saved every epoch, stores `best_val_si_sdri`); resume prefers latest | `snn_finetune.py` |
| 20 | No self-resubmit — manual sbatch needed after each job | `run_train.slurm` runs `sbatch "$0"` if epoch < n_epochs | `run_train.slurm` |
| 21 | `--gres=gpu:1` rejected by cluster | Removed (partition implies GPU) | `run_train.slurm` |
| 22 | `--mem=XG` not supported on rtx4060ti16g | Removed | `run_train.slurm` |
| 23 | Windows CRLF line endings: sbatch rejects `\r\n` | `sed -i 's/\r//' run_train.slurm` after every scp | workflow |

## Fixed — crash-loop (2026-03-28)

| # | Symptom | Fix | File |
|---|---|---|---|
| 24 | Infinite resubmit loop (crash at ep3, resubmit, repeat) | `vl["wform_mean"]` raised `KeyError` when val_every=5 (vl={} at ep3). Changed to `vl.get("wform_mean", float("nan"))` | `snn_finetune.py:~623` |

## Fixed — LibriMix generation (2026-04-30)

| # | Symptom | Fix | File |
|---|---|---|---|
| 25 | `soundfile.LibsndfileError: System error` writing noise wav at 0/3000 — all three generation jobs crashed before writing any files | `write_noise()` called `sf.write()` without creating the `noise/` subdirectory first. `ProcessPoolExecutor` race condition: multiple workers hit the missing dir simultaneously. Fix: `awk` patch inserting `os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)` before `sf.write()`. Also deleted empty stale `dev/` and `test/` dirs (script skips splits when dir exists). | `LibriMix/scripts/create_librimix_from_metadata.py:~358` |

## Fixed — v9 double augmentation (2026-05-05)

| # | Symptom | Fix | File |
|---|---|---|---|
| 26 | v9 train SI-SDRi collapsed to -1.27 dB despite narrower gap. `librimix_dataset.py` already contains `spike_safe_augment()` applying ±6 dB GLOBAL gain jitter. v9's training loop added INDEPENDENT per-source ±6 dB on top → effective ±12 dB SNR swings per utterance. | Reduced `gain_aug_db` from 6.0 to 3.0 in v10/v11 (safe on top of dataset's global jitter, combined max ±9 dB). | `two_speaker_train_v10.py`, `two_speaker_train_v11.py` |

## Fixed — v10 startup logic (never fires on SLURM) (2026-05-12)

| # | Symptom | Fix | File |
|---|---|---|---|
| 27 | SLURM always passes `--resume` flag, so `cfg.resume=True` from the very first job (no checkpoint yet). Starter weight loading (`if starter_path.exists() and not cfg.resume`) and freeze logic (`if freeze_epochs > 0 and not cfg.resume`) were both gated on `not cfg.resume` → neither ever fired. Model trained from random weights with unfrozen encoder. | Restructured startup: (1) resume block runs first, sets `_actually_resumed` flag; (2) starter loading gated on `not _actually_resumed`; (3) EMA initialised AFTER starter weights load; (4) freeze conditioned on `start_epoch <= freeze_epochs` (not `not cfg.resume`). | `two_speaker_train_v10.py`, `two_speaker_train_v11.py` |
| 28 | EMA shadow initialised from random model weights before starter loading code (which never fired due to bug #27). EMA tracked the wrong weights from epoch 1. | Moved EMA initialisation to after starter weight loading. On resume, load EMA state from checkpoint after init. | `two_speaker_train_v10.py`, `two_speaker_train_v11.py` |

## Fixed — v10 performance (2026-05-13)

| # | Symptom | Fix | File |
|---|---|---|---|
| 29 | StatefulSNNSeparator: ~20 s/batch observed on ARC (expected 3-5 s). Root cause: 40 outer (chunks) × 100 inner (LIF time steps) × 6 layers = 24,000 Python function calls per forward pass on batch=8. GPU idle between each Python dispatch. v7 processed B×80=640 samples per inner iteration — 80× more GPU work per Python call. At 20 s/batch × 1,737 batches = ~35,000 s/epoch. Job cancelled. | Replaced StatefulSNNSeparator with StatefulGRUSeparator in v11. cuDNN GRU processes full (B, 100, 256) chunk in one fused kernel. Only 40 Python iterations remain (outer chunk loop). | `sep_model_v10.py`, `two_speaker_train_v11.py` |
| 30 | `rate_t` device mismatch when `spike_recs=[]` (GRU returns no spikes). `_spike_rate_loss([])` returns `torch.tensor(0.0)` on CPU. Adding CPU tensor to CUDA `si_sdr_loss` raises RuntimeError. | Changed `rate_t` assignment to `.to(si_sdr_loss.device)` in `pit_forward`. | `two_speaker_train_v11.py` |

---

## Remaining / future

| # | Issue | Workaround / Fix needed |
|---|---|---|
| R1 | Inference silence threshold too low: `SILENCE_THRESHOLD=1e-3` won't filter non-separated channels | Use relative threshold (5% of mixture RMS). Only matters after model reaches +3 dB |
| R2 | Periodic checkpoints `snn_ep*.pt` missing `model_cfg` | Must manually set arch flags if resuming from periodic checkpoint |
| R3 | `train-360` generation fails: WHAM noise `tr/` files on BeeGFS return `System error` on read (different from the write bug fixed in #25 — this is a read error in `read_sources()` at line ~249) | Use `train-100` (13,900 mixtures). Do not attempt to regenerate `train-360` without first verifying WHAM file integrity on a compute node. |
