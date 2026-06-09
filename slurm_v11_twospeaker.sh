#!/bin/bash
#SBATCH --job-name=snn_v11
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_v11_%j.out
#SBATCH --error=slurm_v11_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
#
# v11 changes vs v10:
#   - StatefulGRUSeparator replaces StatefulSNNSeparator
#       Single cuDNN GRU kernel per chunk vs 100 Python LIF iterations
#       Expected: 2000-4000 s/epoch (vs ~35000 s for v10)
#   - lambda_rate = 0.0 (no spikes to regularise)
#   - Partial weight transfer: encoder + decoder from checkpoints_v7/best_2spk.pt
#   - Everything else identical to v10 (hidden=512, n_layers=6, freeze_epochs=20)
#   - Checkpoint dir: ./checkpoints_v11

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v11

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v11.py \
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
    echo "[slurm] Check slurm_v11_${SLURM_JOB_ID}.err for the traceback."
fi
