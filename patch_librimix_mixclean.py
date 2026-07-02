"""
patch_librimix_mixclean.py — let LibriMix generate mix_clean WITHOUT WHAM noise.

Why: Libri2Mix train-360 metadata references speed-augmented noise files
(`tr/...sp08.wav`, `...sp12.wav`) that LibriMix's `augment_train_noise.py` would
create. Those were never generated on this cluster, so `read_sources()` in
`create_librimix_from_metadata.py` throws on the missing noise file — even for
`--types mix_clean`, because the noise is read unconditionally before the per-type
loop. But `mix_clean` *discards* the noise (`sources_to_mix = transformed_sources[:n_src]`),
so we don't need it at all.

This patch makes two surgical edits to create_librimix_from_metadata.py:
  (1) read_sources: wrap the noise `sf.read` in try/except → silent placeholder
      if the file is missing. (If the noise file DOES exist, behaviour is unchanged.)
  (2) process_utterance: only call `write_noise` when the 'noise' subdir is actually
      requested, so a `--types mix_clean` run never needs/creates a noise/ dir.

mix_clean / s1 / s2 output is byte-identical to a normal run — only a file that
would have been read-then-thrown-away is skipped. Idempotent; writes a .bak once.

Usage (on ARC, with the conda python, from the LibriMix repo root):
    /mnt/beegfs/juashik/.conda/envs/snn/bin/python patch_librimix_mixclean.py
    # or pass an explicit path:
    python patch_librimix_mixclean.py scripts/create_librimix_from_metadata.py
"""

import os
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scripts/create_librimix_from_metadata.py"
if not os.path.isfile(p):
    sys.exit(f"[patch] file not found: {p}\n"
             f"        run this from the LibriMix repo root, or pass the path explicitly.")

src = open(p, encoding="utf-8").read()

old1 = ("    noise_path = os.path.join(wham_dir, row['noise_path'])\n"
        "    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)")
new1 = ("    noise_path = os.path.join(wham_dir, row['noise_path'])\n"
        "    try:\n"
        "        noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)\n"
        "    except Exception:\n"
        "        noise = np.zeros(max(max_length, 1), dtype='float32')")

old2 = ("    # Write the noise and get its path\n"
        "    abs_noise_path = write_noise(mix_id, transformed_sources, dir_path,\n"
        "                                 freq)")
new2 = ("    # Write the noise and get its path\n"
        "    if 'noise' in subdirs:\n"
        "        abs_noise_path = write_noise(mix_id, transformed_sources, dir_path,\n"
        "                                     freq)\n"
        "    else:\n"
        "        abs_noise_path = None")

if new1 in src and new2 in src:
    print("[patch] already applied — nothing to do")
    sys.exit(0)

c1, c2 = src.count(old1), src.count(old2)
if c1 != 1:
    sys.exit(f"[patch] ABORT: noise-read anchor matched {c1}x (expected 1). "
             f"File not modified. Paste lines ~247-249 so the anchor can be fixed.")
if c2 != 1:
    sys.exit(f"[patch] ABORT: write_noise anchor matched {c2}x (expected 1). "
             f"File not modified. Paste the 'Write the noise' block so the anchor can be fixed.")

shutil.copy(p, p + ".bak")
open(p, "w", encoding="utf-8").write(src.replace(old1, new1).replace(old2, new2))
print(f"[patch] OK — patched {p}")
print(f"[patch] backup saved at {p}.bak")
print("[patch] verify:  grep -n -A4 'Read the noise' " + p)
