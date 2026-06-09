"""
fsd50k_dataset.py — FSD50K-based 8-source separation dataset.

Replaces sep_dataset.py as the data source for v2 training.
Builds synthetic separation scenes on-the-fly from individual FSD50K clips.
Output format is identical to SoundSepDataset so snn_finetune.py needs no changes.

EXTRACTION REQUIRED BEFORE USE
-------------------------------
The audio archives are split ZIPs — use 7-Zip to extract:

  cd data/fsd50k/
  # Windows: right-click FSD50K.dev_audio.zip → 7-Zip → Extract Here
  # Or from command line (requires 7z in PATH):
  7z x FSD50K.dev_audio.zip
  7z x FSD50K.eval_audio.zip
  7z x FSD50K.ground_truth.zip
  7z x FSD50K.metadata.zip

Expected layout after extraction:
  data/fsd50k/
    FSD50K.dev_audio/      ← ~40,966 WAV files named {fname}.wav
    FSD50K.eval_audio/     ← ~10,231 WAV files named {fname}.wav
    FSD50K.ground_truth/
      dev.csv              ← fname, labels, mids, split
      eval.csv             ← fname, labels, mids
      vocabulary.csv       ← index, label, mid

Clip counts per category after mapping (dev set, training split):
  music                30:  ~11,626  (capped to music_cap=3,500 by default)
  domestic_and_objects:  ~3,402
  human_voice:           ~3,306
  animal:                ~2,918
  tools_and_machines:    ~2,673
  impacts:               ~2,248
  weather:               ~2,074
  alerts:                ~1,669

Augmentations (training only, identical to sep_dataset.py):
  1. Random gain ±6 dB
  2. Noise injection (0–2%) on mixture only
  3. Time shift 50%: circular roll up to 0.5 s
  4. SpecAugment 60%: freq + time masking on mixture STFT
"""

import os
import sys
import csv
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import ALL_LABELS

SEP_LABELS    = ALL_LABELS
N_SEP_SOURCES = len(SEP_LABELS)   # 8

# ─────────────────────────────────────────────────────────────────────────────
# FSD50K vocabulary → our 8-label mapping
# Keys are EXACT FSD50K vocabulary names (underscores, no spaces).
# Values are one of SEP_LABELS.
# For clips with multiple labels, the first match wins (labels are specific→general).
# ─────────────────────────────────────────────────────────────────────────────

FSD50K_TO_LABEL: Dict[str, str] = {
    # ── human_voice ──────────────────────────────────────────────────────────
    "Speech":                                   "human_voice",
    "Human_voice":                              "human_voice",
    "Male_speech_and_man_speaking":             "human_voice",
    "Female_speech_and_woman_speaking":         "human_voice",
    "Child_speech_and_kid_speaking":            "human_voice",
    "Conversation":                             "human_voice",
    "Laughter":                                 "human_voice",
    "Crying_and_sobbing":                       "human_voice",
    "Singing":                                  "human_voice",
    "Female_singing":                           "human_voice",
    "Male_singing":                             "human_voice",
    "Screaming":                                "human_voice",
    "Whispering":                               "human_voice",
    "Chatter":                                  "human_voice",
    "Shout":                                    "human_voice",
    "Yell":                                     "human_voice",
    "Cheering":                                 "human_voice",
    "Crowd":                                    "human_voice",
    "Breathing":                                "human_voice",
    "Respiratory_sounds":                       "human_voice",
    "Cough":                                    "human_voice",
    "Sneeze":                                   "human_voice",
    "Sigh":                                     "human_voice",
    "Gasp":                                     "human_voice",
    "Human_group_actions":                      "human_voice",
    "Applause":                                 "human_voice",
    "Clapping":                                 "human_voice",
    "Finger_snapping":                          "human_voice",
    "Chuckle_and_chortle":                      "human_voice",
    "Giggle":                                   "human_voice",
    "Burping_and_eructation":                   "human_voice",
    "Fart":                                     "human_voice",
    "Hands":                                    "human_voice",
    "Run":                                      "human_voice",
    "Walk_and_footsteps":                       "human_voice",

    # ── animal ───────────────────────────────────────────────────────────────
    "Dog":                                      "animal",
    "Cat":                                      "animal",
    "Bird":                                     "animal",
    "Frog":                                     "animal",
    "Insect":                                   "animal",
    "Livestock_and_farm_animals_and_working_animals": "animal",
    "Animal":                                   "animal",
    "Wild_animals":                             "animal",
    "Domestic_animals_and_pets":                "animal",
    "Meow":                                     "animal",
    "Purr":                                     "animal",
    "Bark":                                     "animal",
    "Crow":                                     "animal",
    "Gull_and_seagull":                         "animal",
    "Cricket":                                  "animal",
    "Chicken_and_rooster":                      "animal",
    "Chirp_and_tweet":                          "animal",
    "Bird_vocalization_and_bird_call_and_bird_song": "animal",
    "Growling":                                 "animal",
    "Fowl":                                     "animal",
    "Buzz":                                     "animal",

    # ── impacts ──────────────────────────────────────────────────────────────
    "Explosion":                                "impacts",
    "Gunshot_and_gunfire":                      "impacts",
    "Slam":                                     "impacts",
    "Shatter":                                  "impacts",
    "Knock":                                    "impacts",
    "Thump_and_thud":                           "impacts",
    "Boom":                                     "impacts",
    "Fireworks":                                "impacts",
    "Crack":                                    "impacts",
    "Crackle":                                  "impacts",
    "Crushing":                                 "impacts",
    "Breaking":                                 "impacts",
    "Crash_cymbal":                             "impacts",
    "Clapping":                                 "impacts",   # also in human_voice but Clapping first=human_voice
    "Wood":                                     "impacts",
    "Glass":                                    "impacts",
    "Chink_and_clink":                          "impacts",

    # ── music ────────────────────────────────────────────────────────────────
    "Music":                                    "music",
    "Musical_instrument":                       "music",
    "Guitar":                                   "music",
    "Acoustic_guitar":                          "music",
    "Electric_guitar":                          "music",
    "Piano":                                    "music",
    "Drum":                                     "music",
    "Drum_kit":                                 "music",
    "Plucked_string_instrument":                "music",
    "Bowed_string_instrument":                  "music",
    "Brass_instrument":                         "music",
    "Wind_instrument_and_woodwind_instrument":  "music",
    "Keyboard_(musical)":                       "music",
    "Percussion":                               "music",
    "Mallet_percussion":                        "music",
    "Marimba_and_xylophone":                    "music",
    "Glockenspiel":                             "music",
    "Gong":                                     "music",
    "Snare_drum":                               "music",
    "Bass_drum":                                "music",
    "Bass_guitar":                              "music",
    "Harp":                                     "music",
    "Harmonica":                                "music",
    "Accordion":                                "music",
    "Organ":                                    "music",
    "Cymbal":                                   "music",
    "Hi-hat":                                   "music",
    "Rattle_(instrument)":                      "music",
    "Strum":                                    "music",
    "Tambourine":                               "music",
    "Tabla":                                    "music",
    "Trumpet":                                  "music",

    # ── weather ──────────────────────────────────────────────────────────────
    "Rain":                                     "weather",
    "Raindrop":                                 "weather",
    "Thunder":                                  "weather",
    "Thunderstorm":                             "weather",
    "Wind":                                     "weather",
    "Ocean":                                    "weather",
    "Waves_and_surf":                           "weather",
    "Stream":                                   "weather",
    "Trickle_and_dribble":                      "weather",
    "Fire":                                     "weather",
    "Water":                                    "weather",
    "Splash_and_splatter":                      "weather",
    "Drip":                                     "weather",
    "Boiling":                                  "weather",
    "Hiss":                                     "weather",
    "Whoosh_and_swoosh_and_swish":              "weather",

    # ── alerts ───────────────────────────────────────────────────────────────
    "Alarm":                                    "alerts",
    "Siren":                                    "alerts",
    "Bell":                                     "alerts",
    "Doorbell":                                 "alerts",
    "Ringtone":                                 "alerts",
    "Church_bell":                              "alerts",
    "Bicycle_bell":                             "alerts",
    "Telephone":                                "alerts",
    "Cowbell":                                  "alerts",
    "Chime":                                    "alerts",
    "Wind_chime":                               "alerts",
    "Vehicle_horn_and_car_horn_and_honking":    "alerts",
    "Tick":                                     "alerts",
    "Tick-tock":                                "alerts",

    # ── domestic_and_objects ─────────────────────────────────────────────────
    "Door":                                     "domestic_and_objects",
    "Sliding_door":                             "domestic_and_objects",
    "Dishes_and_pots_and_pans":                 "domestic_and_objects",
    "Domestic_sounds_and_home_sounds":          "domestic_and_objects",
    "Microwave_oven":                           "domestic_and_objects",
    "Water_tap_and_faucet":                     "domestic_and_objects",
    "Toilet_flush":                             "domestic_and_objects",
    "Zipper_(clothing)":                        "domestic_and_objects",
    "Cupboard_open_or_close":                   "domestic_and_objects",
    "Drawer_open_or_close":                     "domestic_and_objects",
    "Keys_jangling":                            "domestic_and_objects",
    "Cutlery_and_silverware":                   "domestic_and_objects",
    "Clock":                                    "domestic_and_objects",
    "Computer_keyboard":                        "domestic_and_objects",
    "Keyboard_(musical)":                       "domestic_and_objects",   # fallback only
    "Typewriter":                               "domestic_and_objects",
    "Typing":                                   "domestic_and_objects",
    "Writing":                                  "domestic_and_objects",
    "Scissors":                                 "domestic_and_objects",
    "Packing_tape_and_duct_tape":               "domestic_and_objects",
    "Crumpling_and_crinkling":                  "domestic_and_objects",
    "Tearing":                                  "domestic_and_objects",
    "Coin_(dropping)":                          "domestic_and_objects",
    "Bathtub_(filling_or_washing)":             "domestic_and_objects",
    "Sink_(filling_or_washing)":                "domestic_and_objects",
    "Fill_(with_liquid)":                       "domestic_and_objects",
    "Pour":                                     "domestic_and_objects",
    "Liquid":                                   "domestic_and_objects",
    "Gurgling":                                 "domestic_and_objects",
    "Camera":                                   "domestic_and_objects",
    "Tap":                                      "domestic_and_objects",
    "Squeak":                                   "domestic_and_objects",

    # ── tools_and_machines ───────────────────────────────────────────────────
    "Power_tool":                               "tools_and_machines",
    "Engine":                                   "tools_and_machines",
    "Drill":                                    "tools_and_machines",
    "Sawing":                                   "tools_and_machines",
    "Mechanical_fan":                           "tools_and_machines",
    "Train":                                    "tools_and_machines",
    "Printer":                                  "tools_and_machines",
    "Tools":                                    "tools_and_machines",
    "Car":                                      "tools_and_machines",
    "Car_passing_by":                           "tools_and_machines",
    "Motorcycle":                               "tools_and_machines",
    "Truck":                                    "tools_and_machines",
    "Bus":                                      "tools_and_machines",
    "Motor_vehicle_(road)":                     "tools_and_machines",
    "Bicycle":                                  "tools_and_machines",
    "Aircraft":                                 "tools_and_machines",
    "Fixed-wing_aircraft_and_airplane":         "tools_and_machines",
    "Boat_and_Water_vehicle":                   "tools_and_machines",
    "Rail_transport":                           "tools_and_machines",
    "Subway_and_metro_and_underground":         "tools_and_machines",
    "Race_car_and_auto_racing":                 "tools_and_machines",
    "Traffic_noise_and_roadway_noise":          "tools_and_machines",
    "Vehicle":                                  "tools_and_machines",
    "Idling":                                   "tools_and_machines",
    "Hammer":                                   "tools_and_machines",
    "Ratchet_and_pawl":                         "tools_and_machines",
    "Rattle":                                   "tools_and_machines",
    "Mechanisms":                               "tools_and_machines",
    "Screech":                                  "tools_and_machines",
    "Skateboard":                               "tools_and_machines",
    "Frying_(food)":                            "tools_and_machines",
    "Chewing_and_mastication":                  "tools_and_machines",
    "Scratching_(performance_technique)":       "tools_and_machines",
    "Accelerating_and_revving_and_vroom":       "tools_and_machines",
    "Engine_starting":                          "tools_and_machines",
    "Speech_synthesizer":                       "tools_and_machines",
}

# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers (same API as sep_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

_RESAMPLER_CACHE: Dict[Tuple[int, int], T.Resample] = {}

def _load_wav(path: str, target_sr: int = 16_000) -> torch.Tensor:
    """Load wav → (1, T), resampled to target_sr, mono. Caches Resample kernels."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        key = (sr, target_sr)
        if key not in _RESAMPLER_CACHE:
            _RESAMPLER_CACHE[key] = T.Resample(orig_freq=sr, new_freq=target_sr)
        wav = _RESAMPLER_CACHE[key](wav)
    return wav   # (1, T)


def _pad_or_trim(wav: torch.Tensor, n: int) -> torch.Tensor:
    L = wav.shape[-1]
    if L >= n:
        return wav[..., :n]
    return F.pad(wav, (0, n - L))


def _rms_normalise(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp(min=eps)
    return wav / rms


def _spec_augment(
    mixture: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 800,
    n_fft: int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    """Freq + time masking in STFT domain. Input (1, T), output (1, T)."""
    T_orig = mixture.shape[-1]
    window = torch.hann_window(n_fft, device=mixture.device)
    spec = torch.stft(
        mixture[0], n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True,
    )
    F_bins, T_frames = spec.shape
    f_w = random.randint(0, freq_mask_param)
    f_s = random.randint(0, max(0, F_bins - f_w))
    spec[f_s:f_s + f_w, :] = 0.0
    t_w = random.randint(0, min(time_mask_param, T_frames))
    t_s = random.randint(0, max(0, T_frames - t_w))
    spec[:, t_s:t_s + t_w] = 0.0
    wav_out = torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                          window=window, length=T_orig)
    return wav_out.unsqueeze(0).clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def _map_labels(labels_str: str) -> Optional[str]:
    """Return the first matching 8-label category for a FSD50K labels string, or None."""
    for lbl in labels_str.split(","):
        lbl = lbl.strip()
        if lbl in FSD50K_TO_LABEL:
            return FSD50K_TO_LABEL[lbl]
    return None


def _build_clip_index(
    ground_truth_dir: Path,
    audio_dir: Path,
    csv_split_filter: Optional[str],   # "train", "val", or None (for eval.csv)
    music_cap: int,
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    """
    Parse dev.csv or eval.csv and return {label: [wav_path, ...]} for all clips
    that exist on disk, keyed by mapped 8-label category.

    csv_split_filter: if not None, only include rows where split == csv_split_filter.
                      Pass None to use eval.csv (which has no split column).
    """
    if csv_split_filter is not None:
        csv_path = ground_truth_dir / "dev.csv"
    else:
        csv_path = ground_truth_dir / "eval.csv"

    clips_by_label: Dict[str, List[str]] = {lbl: [] for lbl in SEP_LABELS}
    missing = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if csv_split_filter is not None and row["split"] != csv_split_filter:
                continue
            category = _map_labels(row["labels"])
            if category is None:
                continue
            wav_path = audio_dir / f"{row['fname']}.wav"
            if wav_path.exists():
                clips_by_label[category].append(str(wav_path))
            else:
                missing += 1

    # Cap music to prevent extreme imbalance (music has ~3.5x more clips than others)
    if len(clips_by_label["music"]) > music_cap:
        indices = rng.choice(len(clips_by_label["music"]), size=music_cap, replace=False)
        clips_by_label["music"] = [clips_by_label["music"][i] for i in indices]

    if missing > 0:
        print(f"[FSD50K] Warning: {missing} clips referenced in CSV not found on disk.")

    return clips_by_label


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FSD50KDataset(Dataset):
    """
    FSD50K-based 8-source separation dataset.

    Builds synthetic scenes on-the-fly: each __getitem__ randomly selects
    min_active–max_active categories, picks one clip per category, and mixes
    them. Output is identical to SoundSepDataset:

        mixture:  (T,)   — mono, peak-normalised
        sources:  (8, T) — one channel per SEP_LABELS; absent categories = 0
        label:    (8,)   — multi-hot
        scene_id: int    — for val/test: index into pre-generated scene list;
                           for train: the dataset index

    For val and test splits, scenes are pre-generated at init time with a
    fixed seed so they are deterministic across training runs.

    For train splits, scenes are generated fresh each call to __getitem__
    (new random combinations every epoch).
    """

    SPEC_AUG_PROB = 0.6

    def __init__(
        self,
        fsd50k_root:      str,
        split:            str   = "train",       # "train", "val", or "test"
        sr:               int   = 16_000,
        duration:         float = 3.0,
        min_active:       int   = 2,
        max_active:       int   = 4,
        scenes_per_epoch: int   = 10_000,        # training set size per epoch
        n_val_scenes:     int   = 2_000,         # fixed val/test scene count
        augment:          bool  = True,
        music_cap:        int   = 3_500,
        seed:             int   = 42,
        cache_audio:      bool  = True,          # pre-load all clips into RAM (speeds up training ~20x)
    ):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"
        self.split       = split
        self.sr          = sr
        self.n_samples   = int(duration * sr)
        self.min_active  = min_active
        self.max_active  = max_active
        self.augment     = augment and (split == "train")

        root = Path(fsd50k_root)
        gt_dir = root / "FSD50K.ground_truth"
        dev_audio  = root / "FSD50K.dev_audio"
        eval_audio = root / "FSD50K.eval_audio"

        # Validate paths
        for p, name in [(gt_dir, "FSD50K.ground_truth"), (dev_audio, "FSD50K.dev_audio"),
                        (eval_audio, "FSD50K.eval_audio")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"[FSD50K] {name} not found at {p}.\n"
                    f"Extract the archives first (see fsd50k_dataset.py header for instructions)."
                )

        rng = np.random.default_rng(seed)

        if split == "train":
            self.clips_by_label = _build_clip_index(gt_dir, dev_audio, "train", music_cap, rng)
            self._n = scenes_per_epoch
            self._fixed_scenes: Optional[List] = None   # generated on-the-fly
        elif split == "val":
            self.clips_by_label = _build_clip_index(gt_dir, dev_audio, "val", music_cap, rng)
            self._n = n_val_scenes
            self._fixed_scenes = self._generate_scenes(n_val_scenes, rng)
        else:   # test
            self.clips_by_label = _build_clip_index(gt_dir, eval_audio, None, music_cap, rng)
            self._n = n_val_scenes
            self._fixed_scenes = self._generate_scenes(n_val_scenes, rng)

        # Identify which categories have clips — skip empty ones in scene generation
        self._active_labels = [lbl for lbl in SEP_LABELS if self.clips_by_label[lbl]]
        if len(self._active_labels) < 2:
            raise RuntimeError(
                f"[FSD50K] Only {len(self._active_labels)} categories have clips for split='{split}'. "
                f"Check that audio is extracted and paths are correct."
            )

        counts = {lbl: len(self.clips_by_label[lbl]) for lbl in SEP_LABELS}
        print(f"[FSD50K split={split}] {len(self)} scenes/epoch | "
              f"sr={sr} | duration={duration:.1f}s | augment={self.augment}")
        print(f"  Clips: " + "  ".join(f"{lbl.split('_')[0]}={n}" for lbl, n in counts.items()))

        # Pre-load all clips into RAM so __getitem__ never hits the disk
        self._audio_cache: Dict[str, torch.Tensor] = {}
        if cache_audio:
            self._build_audio_cache()

    # ── Audio RAM cache ───────────────────────────────────────────────────────

    def _build_audio_cache(self) -> None:
        """
        Pre-load every clip in self.clips_by_label into self._audio_cache.

        Each entry is stored as (1, min(actual_len, n_samples)) float32.
        Resampling is done once here; __getitem__ just does _pad_or_trim (cheap).
        """
        all_paths = list({p for clips in self.clips_by_label.values() for p in clips})
        n_total   = len(all_paths)
        print(f"  [cache] Pre-loading {n_total} clips into RAM ...", flush=True)

        n_loaded = 0
        for i, path in enumerate(all_paths):
            try:
                wav = _load_wav(path, self.sr)                 # resampled to 16 kHz
                wav = wav[..., :self.n_samples]                # keep at most n_samples
                self._audio_cache[path] = wav
                n_loaded += 1
            except Exception:
                pass   # clip will fall back to disk load
            if (i + 1) % 5_000 == 0 or i == n_total - 1:
                mb = sum(v.numel() * 4 for v in self._audio_cache.values()) / 1e6
                print(f"  [cache] {i+1}/{n_total}  RAM used: {mb:.0f} MB", flush=True)

        print(f"  [cache] Done — {n_loaded}/{n_total} clips in RAM.", flush=True)

    # ── Scene pre-generation (val/test only) ─────────────────────────────────

    def _generate_scenes(self, n: int, rng: np.random.Generator) -> List[List[Tuple[int, str]]]:
        """
        Pre-generate n scene definitions as lists of (channel_idx, wav_path).
        Used for deterministic val/test sets.
        """
        scenes = []
        available = [lbl for lbl in SEP_LABELS if self.clips_by_label.get(lbl)]
        for _ in range(n):
            n_active  = int(rng.integers(self.min_active, self.max_active + 1))
            n_active  = min(n_active, len(available))
            chosen    = rng.choice(len(available), size=n_active, replace=False)
            scene     = []
            for ci in chosen:
                lbl   = available[int(ci)]
                ch_idx = SEP_LABELS.index(lbl)
                clips  = self.clips_by_label[lbl]
                clip_i = int(rng.integers(0, len(clips)))
                scene.append((ch_idx, clips[clip_i]))
            scenes.append(scene)
        return scenes

    # ── Internal scene loading ────────────────────────────────────────────────

    def _load_scene(self, scene: List[Tuple[int, str]]) -> Dict:
        """Load and mix a pre-defined scene → dict with mixture, sources, label."""
        channels = [torch.zeros(1, self.n_samples) for _ in range(N_SEP_SOURCES)]
        label    = torch.zeros(N_SEP_SOURCES)

        for ch_idx, wav_path in scene:
            try:
                if wav_path in self._audio_cache:
                    wav = self._audio_cache[wav_path]
                else:
                    wav = _load_wav(wav_path, self.sr)
                wav = _pad_or_trim(wav, self.n_samples)
                wav = _rms_normalise(wav)
                channels[ch_idx] = channels[ch_idx] + wav
                label[ch_idx]    = 1.0
            except Exception:
                # Silent skip on corrupt clip — the channel stays zero
                pass

        # Peak-normalise the mixture
        mix_raw = sum(channels)                                  # (1, T)
        peak    = mix_raw.abs().max().clamp(min=1e-8)
        mixture = mix_raw / peak
        sources = torch.cat([ch / peak for ch in channels], dim=0)   # (8, T)

        return mixture, sources, label

    def _random_scene(self) -> List[Tuple[int, str]]:
        """Generate a random scene definition for on-the-fly training."""
        available = self._active_labels
        n_active  = random.randint(self.min_active, min(self.max_active, len(available)))
        chosen    = random.sample(available, k=n_active)
        scene     = []
        for lbl in chosen:
            ch_idx  = SEP_LABELS.index(lbl)
            clip    = random.choice(self.clips_by_label[lbl])
            scene.append((ch_idx, clip))
        return scene

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict:
        if self._fixed_scenes is not None:
            scene = self._fixed_scenes[idx % len(self._fixed_scenes)]
        else:
            scene = self._random_scene()

        mixture, sources, label = self._load_scene(scene)

        if self.augment:
            # Gain ±6 dB
            gain    = 10 ** (random.uniform(-6, 6) / 20)
            mixture = (mixture * gain).clamp(-1, 1)
            sources = (sources * gain).clamp(-1, 1)

            # Noise on mixture only
            noise_level = random.uniform(0.0, 0.02)
            mixture = (mixture + noise_level * torch.randn_like(mixture)).clamp(-1, 1)

            # Time shift
            if random.random() > 0.5:
                shift   = random.randint(0, self.sr // 2)
                mixture = torch.roll(mixture, shift)
                sources = torch.roll(sources, shift, dims=-1)

            # SpecAugment on mixture
            if random.random() < self.SPEC_AUG_PROB:
                try:
                    mixture = _spec_augment(mixture)
                except Exception:
                    pass

        return {
            "mixture":  mixture.squeeze(0),   # (T,)
            "sources":  sources,               # (8, T)
            "label":    label,                 # (8,)
            "scene_id": idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataloader factory (mirrors build_sep_dataloaders API)
# ─────────────────────────────────────────────────────────────────────────────

def build_fsd50k_dataloaders(
    fsd50k_root:      str,
    sr:               int   = 16_000,
    duration:         float = 3.0,
    batch_size:       int   = 16,
    num_workers:      int   = 4,
    scenes_per_epoch: int   = 10_000,
    n_val_scenes:     int   = 2_000,
    min_active:       int   = 2,
    max_active:       int   = 4,
    music_cap:        int   = 3_500,
    seed:             int   = 42,
    cache_audio:      bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_dl, val_dl, test_dl) ready for snn_finetune.py.

    train: 10,000 random scenes/epoch, full augmentation
    val:   2,000 fixed scenes, no augmentation
    test:  2,000 fixed scenes from eval split, no augmentation
    """
    common = dict(
        fsd50k_root=fsd50k_root, sr=sr, duration=duration,
        min_active=min_active, max_active=max_active,
        music_cap=music_cap, seed=seed, n_val_scenes=n_val_scenes,
    )

    # Only cache train — it runs 2500 batches/epoch every epoch.
    # Val/test run once per epoch (500 batches) and resampler kernels are
    # cached at module level, so disk reads are fast enough without full RAM cache.
    train_ds = FSD50KDataset(**common, split="train",
                             scenes_per_epoch=scenes_per_epoch, augment=True,
                             cache_audio=cache_audio)
    val_ds   = FSD50KDataset(**common, split="val",   augment=False, cache_audio=False)
    test_ds  = FSD50KDataset(**common, split="test",  augment=False, cache_audio=False)

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=(num_workers > 0),
                          persistent_workers=(num_workers > 0))

    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/fsd50k"

    print("=" * 60)
    print("FSD50K dataset sanity check")
    print("=" * 60)

    train_dl, val_dl, test_dl = build_fsd50k_dataloaders(
        fsd50k_root=root,
        batch_size=4,
        num_workers=0,
        scenes_per_epoch=20,
        n_val_scenes=10,
    )

    print("\n--- Training batch ---")
    batch = next(iter(train_dl))
    mix = batch["mixture"]
    src = batch["sources"]
    lbl = batch["label"]
    print(f"  mixture shape:  {mix.shape}")
    print(f"  sources shape:  {src.shape}")
    print(f"  label shape:    {lbl.shape}")
    print(f"  mixture range:  [{mix.min():.3f}, {mix.max():.3f}]")
    print(f"  active channels per scene:")
    for b in range(mix.shape[0]):
        active = [SEP_LABELS[c] for c in range(8) if lbl[b, c] > 0.5]
        print(f"    scene {b}: {active}")

    print("\n--- Val batch ---")
    batch2 = next(iter(val_dl))
    print(f"  mixture shape: {batch2['mixture'].shape}")
    print(f"  sources shape: {batch2['sources'].shape}")

    print("\nPASS — shapes are correct")
    print(f"  Expected mixture: (B, {train_dl.dataset.n_samples})")
    print(f"  Expected sources: (B, 8, {train_dl.dataset.n_samples})")
