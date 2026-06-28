#!/bin/bash
#SBATCH --job-name=snn_v16
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --output=slurm_v16_%j.out
#SBATCH --error=slurm_v16_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on this cluster.
#
# v16: dynamic mixing + speed perturbation ONLY (augmentation isolation test).
#   - Same dynamic_mix_dataset as v15 (random pairing, SNR ±5 dB, speed 0.95–1.05x)
#   - spike_safe_augment DISABLED (--no_train_augment)
#   - per-source gain_aug DISABLED (--gain_aug_db 0)
#   - v13-sized DPRNN: bn_dim=64, rnn_hidden=128, 6 blocks
#   - warmstart encoder+decoder from checkpoints_v13/best_2spk.pt (EMA shadow)
#   - Tests whether dynamic mixing helps when not buried under extra augmentations

PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

N_EPOCHS=200
BATCH_SIZE=16
NUM_WORKERS=8
VAL_EVERY=5
MAX_WALL=5.75
LIBRIMIX_ROOT=/mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max
CKPT_DIR=./checkpoints_v16

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
    --log_dir ./runs_v16 \
    --csv_path ./v16_train_log.csv \
    --speed_perturb \
    --no_train_augment \
    --gain_aug_db 0

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
    echo "[slurm] Check slurm_v16_${SLURM_JOB_ID}.err for the traceback."
fi
