#!/bin/bash
#SBATCH --job-name=snn_v6
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_v6_%j.out
#SBATCH --error=slurm_v6_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
#
# v6 changes vs v5 ('Balanced-Power'):
#   - RMS-normalised source mixing in dataset (removes loudness shortcut)
#   - Channel energy balance loss lambda=0.10 (prevents speaker collapse)
#   - lambda_recon 5.0 → 10.0  (push recon error from 0.75 toward < 0.10)
#   - Spike dead zone [0.10, 0.20] → [0.10, 0.30] (model operates at ~0.25)
#   - checkpoints isolated to ./checkpoints_v6 (v5 in ./checkpoints_v5 preserved)

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
SCENES_PER_EPOCH=5000
VAL_EVERY=5
MAX_WALL=5.75
FSD50K_ROOT=/mnt/beegfs/juashik/fsd50k
CKPT_DIR=./checkpoints_v6

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --scenes_per_epoch $SCENES_PER_EPOCH \
    --val_every $VAL_EVERY \
    --fsd50k_root $FSD50K_ROOT \
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
    echo "[slurm] Check slurm_v6_${SLURM_JOB_ID}.err for the traceback."
fi
