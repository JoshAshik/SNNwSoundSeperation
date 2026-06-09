#!/bin/bash
#SBATCH --job-name=snn_sep
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

cd ~/SNNwSoundSeperation

# ── Run training (5.75h limit leaves 15min for job overhead) ─────────────────
$PYTHON snn_finetune.py \
    --resume \
    --max_wall_hours 5.75 \
    --n_epochs 100 \
    --batch_size 8 \
    --num_workers 8 \
    --scenes_per_epoch 3000 \
    --val_every 5 \
    --fsd50k_root /mnt/beegfs/juashik/fsd50k

EXIT_CODE=$?

# ── Auto-resubmit if training is not yet complete ────────────────────────────
LATEST=./checkpoints_snn/best_snn_v2_latest.pt
if [ -f "$LATEST" ]; then
    CURRENT_EPOCH=$($PYTHON -c "
import torch
ckpt = torch.load('$LATEST', map_location='cpu', weights_only=False)
print(ckpt.get('epoch', 0))
")
    echo "[slurm] Completed through epoch $CURRENT_EPOCH / 100"

    if [ "$CURRENT_EPOCH" -lt 100 ]; then
        echo "[slurm] Resubmitting job to continue training..."
        sbatch "$0"
    else
        echo "[slurm] Training complete — all 100 epochs done."
    fi
else
    echo "[slurm] No checkpoint found — training may have failed (exit code $EXIT_CODE)."
fi
