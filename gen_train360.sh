#!/bin/bash
#SBATCH --job-name=gen360
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rtx4060ti16g
#SBATCH --output=gen360_%j.out
#SBATCH --error=gen360_%j.err
#
# Generate Libri2Mix train-360 mix_clean ONLY (wav16k, max mode).
#
# WHAM is decoupled by patch_librimix_mixclean.py — the noise read is now
# exception-guarded and mix_clean discards the noise, so the missing speed-
# augmented (sp08/sp12) WHAM files are a no-op.
#
# The LibriMix generator skips any split whose output dir already exists
# (train-100/dev/test), so ONLY train-360 is built; existing data is untouched.
#
# NOTE: generation is NOT cleanly resumable — the per-split skip is by directory,
# not by file. If this job is interrupted, delete the partial dir before retrying:
#   rm -rf /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max/train-360

PY=/mnt/beegfs/juashik/.conda/envs/snn/bin/python
LM=$HOME/SNNwSoundSeperation/LibriMix
LIBRI=/mnt/beegfs/juashik/librimix

echo "[gen] start $(date)"
$PY "$LM/scripts/create_librimix_from_metadata.py" \
    --librispeech_dir $LIBRI/LibriSpeech \
    --wham_dir        $LIBRI/wham_noise \
    --metadata_dir    "$LM/metadata/Libri2Mix" \
    --librimix_outdir $LIBRI \
    --n_src 2 \
    --freqs 16k \
    --modes max \
    --types mix_clean
echo "[gen] exit=$? end $(date)"

OUT=$LIBRI/Libri2Mix/wav16k/max/train-360
for d in mix_clean s1 s2; do
    printf "[gen] %s: " "$d"; ls "$OUT/$d" 2>/dev/null | wc -l
done
