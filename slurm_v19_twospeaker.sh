#!/bin/bash
#SBATCH --job-name=snn_v19
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v19_%j.out
#SBATCH --error=slurm_v19_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v19: the v17 BEST recipe (finer encoder k=16/s=8, from scratch, pure SI-SDR,
# plateau LR) run on the larger REAL dataset train-360 — the DATA lever.
# Capacity was ruled out (v14, v18 both failed/overfit); train-360 is ~3.6x more
# real mixtures from the SAME pipeline as dev/test (no distribution shift).
#
# Reuses two_speaker_train_v17.py UNCHANGED — only data + schedule scale:
#   --train_split train-360   (vs train-100)
#   --n_epochs 90             (train-360 ~6350 batches/ep at batch 8 → ~90 ep gives
#                              ~1.6-2x v17's ~313k updates; full 200 would be wasteful)
#   --batch_size 8           (held at v17's value for a clean data comparison)
#   --val_every 2            (longer epochs → validate more often in epoch terms)
#   --plateau_patience 3     (scaled so the patience window ~matches v17 in updates)
#
# Watch dev: stop (scancel) if it sets no new best for ~15-20 epochs and declines
# (best_2spk.pt already holds the peak). With 3.6x more data, overfitting should
# arrive later than v18.

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=90
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=2
PLATEAU_PATIENCE=3
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v19

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v17.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --train_split train-360 \
    --val_split dev \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --plateau_patience $PLATEAU_PATIENCE \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --log_dir ./runs_v19 \
    --csv_path ./v19_train_log.csv

EXIT_CODE=$?

LATEST=$CKPT_DIR/best_2spk_latest.pt
if [ -f "$LATEST" ]; then
    CURRENT_EPOCH=$($PYTHON -c "
import torch
ckpt = torch.load('$LATEST', map_location='cpu', weights_only=False)
print(ckpt.get('epoch', 0))
")
    echo "[slurm] Completed through epoch $CURRENT_EPOCH / $N_EPOCHS"

    if [ "$CURRENT_EPOCH" -lt "$N_EPOCHS" ]; then
        echo "[slurm] Resubmitting job to continue training..."
        sbatch "$0"
    else
        echo "[slurm] Training complete — all $N_EPOCHS epochs done."
    fi
else
    echo "[slurm] No checkpoint found — training may have failed (exit $EXIT_CODE)."
    echo "[slurm] Check slurm_v19_${SLURM_JOB_ID}.err for the traceback."
fi
