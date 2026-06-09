"""
sort_data.py – Organise a flat data directory into sources/, transients/, dynamic/, shared/
               based on the clip paths listed in original_data_map.json.

Usage:
    # Dry run first (shows what would happen, moves nothing):
    python sort_data.py --src "C:/Users/Josh Ashik/Documents/SNNwSoundSeperation/data" --dry-run

    # Then for real (moves files into subdirectories within --dst):
    python sort_data.py --src "C:/Users/Josh Ashik/Documents/SNNwSoundSeperation/data"

    # Copy instead of move (keeps originals):
    python sort_data.py --src "C:/Users/Josh Ashik/Documents/SNNwSoundSeperation/data" --copy

    # Custom destination:
    python sort_data.py --src /flat/wavs --dst ./data

Result:
    data/
    ├── sources/       ← source_clips WAVs
    ├── transients/    ← transient_clips WAVs
    ├── dynamic/       ← BRIR-processed WAVs
    └── shared/        ← files that appear under multiple clip types (3 files)
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


CLIP_TYPE_TO_SUBDIR = {
    "source":    "sources",
    "transient": "transients",
    "dynamic":   "dynamic",
    "shared":    "shared",
}

SCENE_KEYS = {
    "source_clips":    "source",
    "transient_clips": "transient",
    "dynamic_clips":   "dynamic",
}


def build_file_map(data_map_json):
    """
    Returns {filename: clip_type} for every clip in the data map.
    Files that appear under more than one clip type are assigned 'shared'.
    """
    with open(data_map_json, encoding="utf-8") as f:
        data_map = json.load(f)

    all_types = defaultdict(set)
    for scene in data_map["data"]:
        for key, clip_type in SCENE_KEYS.items():
            for clip in scene.get(key, []):
                fname = Path(clip["path"]).name
                all_types[fname].add(clip_type)

    file_map = {}
    conflicts = []
    for fname, types in all_types.items():
        if len(types) > 1:
            file_map[fname] = "shared"
            conflicts.append((fname, sorted(types)))
        else:
            file_map[fname] = next(iter(types))

    if conflicts:
        print(f"[sort_data] NOTE: {len(conflicts)} file(s) appear under multiple clip types -> shared/")
        for fname, types in conflicts:
            print(f"  {fname}  ({' + '.join(types)})")
    print()

    return file_map


def sort_data(src_dir, dst_dir, data_map_json, dry_run=False, copy=False):
    action = "Copy" if copy else "Move"
    if dry_run:
        print(f"[sort_data] DRY RUN -- no files will be {action.lower()}d\n")

    src_dir       = Path(src_dir)
    dst_dir       = Path(dst_dir)
    data_map_json = Path(data_map_json)

    if not src_dir.exists():
        print(f"[sort_data] ERROR: source directory not found: {src_dir}")
        print(f"  On Windows, quote paths that contain spaces:")
        print(f'  python sort_data.py --src "C:/Users/Your Name/path/to/data"')
        return

    file_map = build_file_map(data_map_json)
    print(f"[sort_data] Data map references {len(file_map)} unique files")

    for subdir in CLIP_TYPE_TO_SUBDIR.values():
        if not dry_run:
            (dst_dir / subdir).mkdir(parents=True, exist_ok=True)

    wav_files = list(src_dir.rglob("*.wav"))
    print(f"[sort_data] Found {len(wav_files)} WAV file(s) in {src_dir}\n")

    if len(wav_files) == 0:
        print("  Nothing to do. Check that --src points to the folder containing your WAV files.")
        return

    counts     = defaultdict(int)
    skipped    = []
    not_in_map = []

    for wav_path in sorted(wav_files):
        fname = wav_path.name

        if fname not in file_map:
            not_in_map.append(fname)
            continue

        clip_type = file_map[fname]
        subdir    = CLIP_TYPE_TO_SUBDIR[clip_type]
        dst_path  = dst_dir / subdir / fname

        if dst_path.exists():
            skipped.append(fname)
            continue

        print(f"  {action}: {fname}  ->  {subdir}/")
        if not dry_run:
            if copy:
                shutil.copy2(wav_path, dst_path)
            else:
                shutil.move(str(wav_path), str(dst_path))

        counts[subdir] += 1

    print(f"\n{'─'*55}")
    label = "[DRY RUN] Would move" if dry_run else "Moved"
    print(f"  {label}:")
    if counts:
        for subdir, n in sorted(counts.items()):
            print(f"    {dst_dir / subdir}:  {n} file(s)")
    else:
        print("    (nothing new to move)")

    if skipped:
        print(f"\n  Already at destination (skipped): {len(skipped)} file(s)")
    if not_in_map:
        print(f"\n  Not referenced in data map (left in place): {len(not_in_map)} file(s)")
        for f in not_in_map[:10]:
            print(f"    {f}")
        if len(not_in_map) > 10:
            print(f"    ... and {len(not_in_map) - 10} more")

    total_placed = sum(counts.values()) + len(skipped)
    missing = len(file_map) - total_placed
    print(f"\n  Coverage: {total_placed}/{len(file_map)} mapped files found")
    if missing > 0:
        print(f"  Warning: {missing} mapped file(s) were not found in {src_dir}")
    else:
        print(f"  All mapped files accounted for")
    print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort flat WAV directory into sources/, transients/, dynamic/, shared/"
    )
    parser.add_argument("--src",     required=True,    help="Folder containing unsorted WAV files")
    parser.add_argument("--dst",     default="./data", help="Destination root (default: ./data)")
    parser.add_argument("--map",     default="original_data_map.json", help="Path to original_data_map.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving anything")
    parser.add_argument("--copy",    action="store_true", help="Copy files instead of moving")
    args = parser.parse_args()

    sort_data(
        src_dir       = args.src,
        dst_dir       = args.dst,
        data_map_json = args.map,
        dry_run       = args.dry_run,
        copy          = args.copy,
    )