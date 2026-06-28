#!/bin/bash
#SBATCH --job-name=snn_v17
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v17_%j.out
#SBATCH --error=slurm_v17_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v17: finer encoder (k=16, s=8) trained FROM SCRATCH + reference DPRNN recipe.
#   A/D  encoder k=16 s=8 (was 32/16), no warmstart, no freeze (end-to-end)
#   B    pure SI-SDR loss: lambda_recon=0, lambda_spec=0.1
#   C    fixed REAL Libri2Mix train-100 (no dynamic mixing), augmentation OFF
#   E    Adam lr=1e-3, ReduceLROnPlateau (halve, patience 4 val-checks), clip 5
#   batch_size=8 (stride-8 doubles activations vs v16's stride-16 batch 16)
#
# Gate before running: oracle_mask_diagnostic.py should show the 16:8 window
# beating the 32:16 window by a meaningful margin (>~1 dB). If not, the encoder
# stride is not the ceiling — revisit before spending this run.

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v17

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
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --log_dir ./runs_v17 \
    --csv_path ./v17_train_log.csv

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
    echo "[slurm] Check slurm_v17_${SLURM_JOB_ID}.err for the traceback."
fi
