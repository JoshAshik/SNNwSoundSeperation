#!/bin/bash
#SBATCH --job-name=snn_v10
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_v10_%j.out
#SBATCH --error=slurm_v10_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
#
# v10 changes vs v9:
#   - StatefulSNNSeparator: LIF state persists across all 40 chunks per 4s clip
#   - hidden 256 → 512, n_layers 3 → 6  (separator capacity ~4×)
#   - freeze_epochs = 20  (fresh separator needs warm-up)
#   - gain_aug_db = 3.0  (reduced from v9's 6.0)
#   - Partial weight transfer: encoder + decoder from checkpoints_v7/best_2spk.pt
#   - Separator freshly initialised (shapes changed)
#   - Checkpoint dir: ./checkpoints_v10
#
# Performance note:
#   StatefulSNNSeparator processes chunks sequentially (batch=B=8 per step).
#   Epochs are expected to be 3-5x slower than v7/v9 (~3000-5000 s/epoch).
#   At ~4000 s/epoch: ~5-6 epochs per 6h job -> ~35-40 jobs for 200 epochs.
#   Self-resubmission handles this automatically.

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v10

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v10.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR

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
    echo "[slurm] Check slurm_v10_${SLURM_JOB_ID}.err for the traceback."
fi
