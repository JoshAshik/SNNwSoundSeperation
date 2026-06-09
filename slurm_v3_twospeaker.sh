#!/bin/bash
#SBATCH --job-name=snn_2spk
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=slurm_2spk_%j.out
#SBATCH --error=slurm_2spk_%j.err
#
# ARC-specific: --gres and --mem are NOT valid on rtx4060ti16g.
# The partition already implies one GPU. Do not add them.
#
# two_speaker_train.py must exist in ~/SNNwSoundSeperation/ before sbatch.
# It should accept: --resume --max_wall_hours --n_epochs --batch_size
#                   --num_workers --scenes_per_epoch --val_every
#                   --fsd50k_root --ckpt_dir

# ── Python env (conda activate does NOT work in Slurm — use full path) ───────
PYTHON=/mnt/beegfs/juashik/.conda/envs/snn/bin/python

# ── Training configuration ────────────────────────────────────────────────────
N_EPOCHS=200
BATCH_SIZE=8
NUM_WORKERS=8
SCENES_PER_EPOCH=5000
VAL_EVERY=5
MAX_WALL=5.75                             # exit 15 min before the 6h wall time
FSD50K_ROOT=/mnt/beegfs/juashik/fsd50k
CKPT_DIR=./checkpoints_2spk              # isolated from v1/v2 checkpoints_snn/

# ── Environment variables ─────────────────────────────────────────────────────

# Reduce CUDA memory fragmentation from variable-length BPTT graphs and
# chunked SNN processing (each chunk allocates/frees intermediate tensors).
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Align PyTorch CPU thread pool with the cpus-per-task allocation.
# Without this, PyTorch spawns os.cpu_count() threads which fight with
# the 8 DataLoader workers for the 4 allocated CPUs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Print a Python-level stack trace if the process receives SIGSEGV or SIGFPE.
# Catches silent crashes from C extensions (e.g. PESQ's C library, snntorch
# surrogate gradient if a future version changes the C backend).
export PYTHONFAULTHANDLER=1

# Prevent glibc from fragmenting memory when the DataLoader workers do
# repeated small-allocation/free cycles while building audio batches.
# 128 MB threshold: chunks above this use mmap (returned to OS on free)
# rather than the brk heap (never returned, causing apparent memory growth).
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=134217728

# ── Run ───────────────────────────────────────────────────────────────────────
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

# ── Auto-resubmit if not yet complete ────────────────────────────────────────
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
    echo "[slurm] Check slurm_2spk_${SLURM_JOB_ID}.err for the traceback."
fi
