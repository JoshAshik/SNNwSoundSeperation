#!/bin/bash
#SBATCH --job-name=snn_v19b
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v19b_%j.out
#SBATCH --error=slurm_v19b_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v19b: WARMSTART RECOVERY of the v19/train-360 run that diverged.
#   The first attempt (slurm_v19 -> two_speaker_train_v17.py on train-360) climbed
#   to val +7.95 (ep26) then diverged to NaN because it ran ~165k steps at the full
#   lr=1e-3 (plateau never decayed it while val kept improving).
#
# two_speaker_train_v19.py recovers WITHOUT redoing the good epochs:
#   - warmstart the full model from checkpoints_v19/best_2spk.pt (the +7.95 weights)
#   - fresh Adam at lr=3e-4 (below the 1e-3 that diverged)  [PRIMARY FIX]
#   - loss computed in fp32 + non-finite-loss batches skipped  [DEFENSIVE]
#   - writes to a FRESH checkpoints_v19b/ so the NaN latest can't be resumed
#
# NOTE: the warmstart source (checkpoints_v19/best_2spk.pt) must still exist and be
# NaN-free (verified: best_val=+7.95). Do NOT delete checkpoints_v19/.

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=90
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=2
PLATEAU_PATIENCE=3
LR=3e-4
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v19b
WARMSTART=./checkpoints_v19/best_2spk.pt

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v19.py \
    --resume \
    --warmstart $WARMSTART \
    --max_wall_hours $MAX_WALL \
    --train_split train-360 \
    --val_split dev \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --plateau_patience $PLATEAU_PATIENCE \
    --lr $LR \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --log_dir ./runs_v19b \
    --csv_path ./v19b_train_log.csv

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
    echo "[slurm] Check slurm_v19b_${SLURM_JOB_ID}.err for the traceback."
fi
