#!/bin/bash
#SBATCH --job-name=snn_v18
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v18_%j.out
#SBATCH --error=slurm_v18_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v18: v17 recipe + WIDER separator bottleneck (Experiment F). Single variable.
#   F    dprnn_bn_dim 64 -> 128 (the only change vs v17)
#   held identical to v17: encoder k=16/s=8 from scratch, no warmstart/freeze,
#        pure SI-SDR loss (lambda_recon=0, lambda_spec=0.1), fixed real train-100,
#        augmentation OFF, Adam lr=1e-3 + ReduceLROnPlateau (ABS 0.01 dB), clip 5
#
# batch_size=16: v17 peaked at 7.65 GB on a 40+ GB A100 at batch 8; the wider
# bottleneck raises memory only modestly, so batch 16 stays well within VRAM and
# uses the A100 efficiently (~halves the batch count vs batch 8). Drop to 8 if a
# smaller GPU is allocated.

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=16
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v18

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v18.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --log_dir ./runs_v18 \
    --csv_path ./v18_train_log.csv

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
    echo "[slurm] Check slurm_v18_${SLURM_JOB_ID}.err for the traceback."
fi
