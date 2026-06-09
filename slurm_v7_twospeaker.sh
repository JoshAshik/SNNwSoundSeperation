#!/bin/bash
#SBATCH --job-name=snn_v7
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_v7_%j.out
#SBATCH --error=slurm_v7_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
#
# v7 changes vs v6:
#   - Architecture: ConvDecoder (3× Conv1d+GroupNorm+PReLU + ConvTranspose1d)
#     replaces single ConvTranspose1d — targets hissing artifact from v6
#   - Dataset: Libri2Mix train-100 (13,900 pre-generated pairs on BeeGFS)
#     replaces FSD50K on-the-fly mixing — aligns with academic 10 dB target
#   - Starting checkpoint: v7_starter.pt (encoder from v6, separator+decoder fresh)
#   - Freeze phase: encoder only (decoder refine blocks must train from epoch 1)
#   - lambda_recon 5.0 (conservative; Libri2Mix sources are clean speech)
#   - Checkpoint dir: ./checkpoints_v7

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v7

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v7.py \
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
    echo "[slurm] Check slurm_v7_${SLURM_JOB_ID}.err for the traceback."
fi
