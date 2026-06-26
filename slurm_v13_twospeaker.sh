#!/bin/bash
#SBATCH --job-name=snn_v13
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_v13_%j.out
#SBATCH --error=slurm_v13_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
#
# v13: DPRNN separator — architecture upgrade from v12.
#   - warmstart encoder+decoder from checkpoints_v12a/best_2spk.pt (EMA shadow)
#   - fresh DPRNNSeparator (separator keys filtered out)
#   - lr=5e-4, freeze_epochs=20, dropout=0.1
#   - dataset augment ON, gain_aug_db=3.0

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v13

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v13.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --train_augment

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
    echo "[slurm] Check slurm_v13_${SLURM_JOB_ID}.err for the traceback."
fi
