# Environment
<!-- dependencies: run_train.slurm -->

## ARC Cluster (NCSU) — PRIMARY TRAINING ENVIRONMENT

```
Host:     arc.csc.ncsu.edu  (login node)
GPUs:     RTX 4060 Ti 16GB  (partition: rtx4060ti16g) — v7–v13
          A100 40/80GB      (partition: a100, node c33) — v14+
RAM:      192 GB per node
Python:   /mnt/beegfs/juashik/.conda/envs/snn/bin/python
Packages: torch 2.5.1+cu121 (supports sm_50–sm_90), snntorch 0.9.4, torchaudio, tensorboard, pesq, mir_eval
Data:     /mnt/beegfs/juashik/librimix/Libri2Mix/wav16k/max/  (BeeGFS)
Code:     ~/SNNwSoundSeperation/
SSH key:  C:\Users\Josh Ashik\private_key.pem  (4096-bit RSA)
```

### GPU compatibility (torch 2.5.1+cu121)
```
Supported:     sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
  RTX 4060 Ti  → sm_89 (Ada Lovelace) — works via sm_86 compat
  A100         → sm_80 — native support
  H100         → sm_90 — native support
NOT supported: sm_120 (Blackwell)
  RTX 5060 Ti  → sm_120 — FAILS (tested: RuntimeError no kernel image)
```

### SSH config (C:\Users\Josh Ashik\.ssh\config)
```
Host arc.csc.ncsu.edu
   HostName arc.csc.ncsu.edu
   User juashik
   IdentityFile "/c/Users/Josh Ashik/private_key.pem"

Host c*
   HostName %h
   User juashik
   ProxyJump arc.csc.ncsu.edu
   IdentityFile "/c/Users/Josh Ashik/private_key.pem"
```

### ARC constraints and pitfalls

| Issue | Rule |
|---|---|
| `--gres=gpu:1` | **INVALID** on this cluster — partition implies GPU, do not add |
| `--mem=XG` | **INVALID** on rtx4060ti16g — not supported, do not add |
| `conda activate` | Doesn't work in Slurm scripts — always use full Python path |
| Windows `\r\n` | sbatch rejects CRLF — run `sed -i 's/\r//' run_train.slurm` after every scp from Windows |
| BeeGFS access | Only accessible from **compute nodes**, not login node |
| `sbatch` location | Must be run from home dir (`~/SNNwSoundSeperation`), NOT from BeeGFS |
| Home quota | 40 GB — code+checkpoints only; all data goes to BeeGFS |
| Wall time | 6 hours per job — `run_train.slurm` self-resubmits at epoch < n_epochs |

### Speed (v13 training — DPRNNSeparator, A100)
```
~160 s/epoch  (2.7 min; no Python loop, all cuDNN BiLSTM)
~200 epochs in ~1 job  (A100; 200 × 160s ≈ 9h, needs 2 SLURM jobs)
```

### Speed (v11 training — StatefulGRUSeparator, rtx4060ti16g)
```
~2000–4000 s/epoch  (expected; 40 GRU kernel calls per forward pass)
~10–20 jobs for 200 epochs  (6h wall, self-resubmitting)
```

### Speed (v7 training — ChunkedSNNSeparator, reference)
```
~1000 s/epoch
~25h total for 200 epochs  (~5 jobs)
```

### Speed (v10 training — StatefulSNNSeparator, ABANDONED)
```
~20 s/batch observed  (~35,000 s/epoch — impractical)
Root cause: 4000 Python-CUDA dispatches per forward pass (40 chunks × 100 LIF steps × B=8)
```

### Speed (v2 training — 8-ch legacy, reference)
```
~1.10s/batch  (no torch.compile)
~412s/epoch   (no-val epochs)
~2 jobs total for 100 epochs
```

---

## Local Windows — inference and code editing only

```
OS:  Windows 11 Enterprise 10.0.22635
GPU: GTX 1660 Super (6 GB VRAM)
```

### Python environments — IMPORTANT

```
python3  → C:\Users\Josh Ashik\AppData\Local\...\python.exe
           Has: torch 2.5.1+cu121, snntorch 0.9.4, torchaudio, tensorboard, pesq
           USE THIS for all SNNwSoundSeperation work

python   → C:\Users\Josh Ashik\Documents\dementiaModelOptimized051925\.venv
           WRONG project venv — missing tensorboard — DO NOT USE
```

### Windows-specific pitfalls

- **Non-ASCII in print()**: always have `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at top of scripts
- **oneDNN warnings**: `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` must be set before any imports
- **num_workers**: previously caused `OSError WinError 1455` (pagefile too small). Fixed: pagefile expanded to 22.5 GB. `num_workers=2` safe locally.
- **torch.compile**: DISABLED permanently — see [bugs.md](bugs.md) #16/#17

---

## Required packages

```
torch, torchaudio, snntorch (0.9.4), mir_eval, pandas, tensorboard, numpy, pesq
```

## Audio format (all files)

```
Mono, 16kHz, 3 seconds = 48,000 samples
```
