#!/bin/bash
#SBATCH --job-name=snn_v15
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v15_%j.out
#SBATCH --error=slurm_v15_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v15: dynamic clean mixing + speed perturbation (v13-sized DPRNN).
#   - on-the-fly random speaker pairing from s1/+s2/ utterance pool
#   - speed perturbation 0.95x–1.05x on each source independently
#   - random relative SNR ±5 dB between sources
#   - warmstart encoder+decoder from checkpoints_v13/best_2spk.pt (EMA shadow)
#   - v13-sized DPRNN: bn_dim=64, rnn_hidden=128, 6 blocks
#   - lr=5e-4, freeze_epochs=20, dropout=0.1, batch_size=16

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=16
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v15

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

cd ~/SNNwSoundSeperation

$PYTHON two_speaker_train_v15.py \
    --resume \
    --max_wall_hours $MAX_WALL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --val_every $VAL_EVERY \
    --librimix_root $LIBRIMIX_ROOT \
    --ckpt_dir $CKPT_DIR \
    --speed_perturb \
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
    echo "[slurm] Check slurm_v15_${SLURM_JOB_ID}.err for the traceback."
fi
