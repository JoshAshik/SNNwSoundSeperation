# CLAUDE.md

Repository handoff notes for coding agents working on this SNN-TasNet audio source separation project.

## Current state

- Active track: 2-speaker Libri2Mix separation with `sep_model_v10.py` and `snn_mode="gru_stateful"`.
- Current best run: v12 Run A, `checkpoints_v12a/best_2spk.pt`.
- Best validation SI-SDRi: **+5.46 dB**, a +0.31 dB improvement over v11 (+5.15 dB).
- Architecture is unchanged from v11; v12 is a stage-2 warmstart fine-tune from the v11 EMA-shadow weights.
- Train SI-SDRi was about +6.2 dB with a ~0.7 dB train/val gap, which is healthy and does not indicate overfitting.

## Final v12 ablation results

| Run | Config | Best Val SI-SDRi | vs v11 (+5.15) | Result |
|-----|--------|------------------|---------------|--------|
| A | `--train_augment` | **+5.46 dB** | **+0.31 dB** | Best checkpoint |
| B | `--no_train_augment` | +5.45 dB | +0.30 dB | Essentially tied with A |
| C | `--no_train_augment --lambda_spec 0.75` | +5.20 dB | +0.05 dB | Underperformed |

Takeaways:

- Dataset augmentation has little effect for stage-2 fine-tuning of converged weights.
- Stage-2 warmstart works, but gains are modest at about +0.3 dB.
- Stronger spectral loss hurt performance; keep the default `lambda_spec=0.5`.

## Important files

- `two_speaker_train_v12.py` - active v12 stage-2 fine-tuning script.
- `sep_model_v10.py` - active model architecture, including `StatefulGRUSeparator`.
- `slurm_v12_twospeaker.sh` - ARC self-resubmitting SLURM job.
- `eval_dataset.py` - batch SI-SDRi evaluation on Libri2Mix splits.
- `two_speaker_inference.py` - single-mixture two-speaker inference.
- `docs/training.md` - detailed training status, history, commands, checkpoints.
- `docs/decisions.md` - rationale and experiment conclusions.

## Checkpoint guardrails

Do not overwrite these checkpoints:

- `checkpoints_v12a/best_2spk.pt` - v12 Run A all-time best, +5.46 dB.
- `checkpoints_v12/best_2spk.pt` - v12 Run B, +5.45 dB.
- `checkpoints_v11/best_2spk.pt` - v11 baseline for v12 warmstarts, +5.15 dB.
- Earlier reference bests noted in `docs/training.md`.

## Common commands

Evaluate the current best checkpoint:

```bash
python3 eval_dataset.py --model checkpoints_v12a/best_2spk.pt \
    --librimix_root /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max \
    --split dev --output_csv eval_dev_v12a.csv
```

Run single-file inference:

```bash
python3 two_speaker_inference.py \
    --model checkpoints_v12a/best_2spk.pt \
    --input mix.wav \
    --out_dir ./pit_output
```

Rerun v12 stage-2 training on ARC if needed:

```bash
cd ~/SNNwSoundSeperation
sbatch slurm_v12_twospeaker.sh
```

## Development notes

- Use EMA-shadow weights for inference and evaluation when present.
- v12 uses `gain_aug_db=0.0`, `freeze_epochs=0`, `lr=1e-5`, and `lambda_spec=0.5`.
- `lambda_rate=0.0` for GRU runs because there are no LIF spikes.
- Keep v10 StatefulSNN as reference only; it was cancelled because it was too slow.
