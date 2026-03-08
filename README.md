# groot-automation

# GR00T Training Automation Pipeline

Automates the full GR00T N1.5 model training workflow — from raw dataset to SLURM training jobs — with a single command.

> **Status:** Phases 1–4 complete and tested. Phases 5–7 in development.

---

## Overview

Previously, training a new dataset required 20–30 minutes of manual work per dataset (calculating splits, writing YAML configs, writing sbatch scripts, submitting jobs). This pipeline reduces that to a **single command in under 30 seconds**.

```
python automating_groot/run_pipeline.py \
  --dataset-path /rlwrld-dataset/.../YOUR_DATASET \
  --dataset-name YOUR_DATASET_NAME \
  --partition rlwrld
```

---

## Pipeline Architecture

```
run_pipeline.py  ←  single entry point
       │
       ├── Phase 1: phase1_split.py       → splits dataset into 10%, 50%, 100%
       ├── Phase 2: phase2_configs.py     → generates YAML training configs
       ├── Phase 3: phase3_train.py       → submits training jobs to SLURM
       ├── Phase 4: phase4_eval.py        
       ├── Phase 5: phase5_results.py     → parses results, calculates success   [coming]
       └── Phase 6: phase6_leaderboard.py → pushes results to leaderboard        [coming]
```

---

## Phases

### ✅ Phase 1 — Dataset Splitting (`phase1_split.py`)
Reads `total_episodes` from the dataset's `meta/info.json` and creates three subsets using `sample_lerobot_dataset_updated.py`.

| Split | Episodes | Output Folder |
|-------|----------|---------------|
| 10%   | ~52      | `{OUTPUT_BASE}/{name}-10pct` |
| 50%   | ~260     | `{OUTPUT_BASE}/{name}-50pct` |
| 100%  | ~520     | `{OUTPUT_BASE}/{name}-100pct` |

Automatically skips a split if the output folder already exists.

---

### ✅ Phase 2 — YAML Config Generation (`phase2_configs.py`)
Generates one YAML training config per split in `gr00t/configs/`.

```yaml
train:
  datasets:
    - path: /rlwrld1/home/yashu/dataset/YOUR_DATASET-10pct
      embodiment_tag: new_embodiment
      data_config: allex_thetwo_ck40_egostereo
      weight: 1.0
```

Automatically skips generation if the YAML already exists.

---

### ✅ Phase 3 — Training Job Submission (`phase3_train.py`)
Generates SBATCH scripts and submits them to SLURM. Each job runs with:

- GPUs: 4
- Batch size: 32
- Learning rate: 1e-4
- Max steps: 30,000
- Save steps: 10,000
- Workers: 8

---

### ⏳ Phase 4 — Checkpoint Detection (`phase4_checkpoints.py`)
Monitors SLURM jobs and detects final checkpoints when training completes. *(In development)*

### ⏳ Phase 5 — Eval Submission (`phase5_eval.py`)
Submits evaluation jobs to Isaac Sim across 5 distances (0cm, 1cm, 3cm, 5cm, 7cm). *(In development)*

### ⏳ Phase 6 — Results Parsing (`phase6_results.py`)
Parses evaluation results JSON and calculates success rates per model per distance. *(In development)*

### ⏳ Phase 7 — Leaderboard Update (`phase7_leaderboard.py`)
Pushes final success rates to the leaderboard. *(In development)*

---

## Requirements

- Python 3.11
- SLURM cluster access
- Virtual environment at `~/rlwrld_isaac/.venv`
- GR00T repo at `~/rlwrld_isaac/gr00t`

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/ys2064/groot-automation
```

### 2. Place files in the gr00t directory
```bash
cp -r groot-automation/ ~/rlwrld_isaac/gr00t/automating_groot/
```

### 3. Activate the virtual environment
```bash
source ~/rlwrld_isaac/.venv/bin/activate
```

---

## Usage

### Run the full pipeline (recommended)
```bash
cd ~/rlwrld_isaac/gr00t

python automating_groot/run_pipeline.py \
  --dataset-path /rlwrld-dataset/foundry-dvc/data/gold/teleop-sim/V4/224/YOUR_DATASET \
  --dataset-name YOUR_DATASET_NAME \
  --partition rlwrld
```

### Run individual phases (for debugging)
```bash
# Phase 1 only
python automating_groot/phase1_split.py \
  --dataset-path /rlwrld-dataset/.../YOUR_DATASET \
  --dataset-name YOUR_DATASET_NAME

# Phase 2 only
python automating_groot/phase2_configs.py \
  --dataset-name YOUR_DATASET_NAME

# Phase 3 only
python automating_groot/phase3_train.py \
  --dataset-name YOUR_DATASET_NAME \
  --partition rlwrld
```

---

## Monitoring

### Check running jobs
```bash
squeue --me
```

### Check training progress
```bash
tail -f ~/rlwrld_isaac/gr00t/log/<JOB_ID>-*.err
```

### Check wandb
Training runs are automatically tracked on [Weights & Biases](https://wandb.ai/yashushukla2014-rlwrld/huggingface).

---

## Output Structure

```
/rlwrld1/home/yashu/
├── dataset/
│   ├── YOUR_DATASET-10pct/     ← Phase 1 output
│   ├── YOUR_DATASET-50pct/
│   └── YOUR_DATASET-100pct/
│
└── output/train/
    ├── YOUR_DATASET_10pct/     ← Phase 3 training output
    │   ├── checkpoint-10000/
    │   ├── checkpoint-20000/
    │   └── checkpoint-30000/
    ├── YOUR_DATASET_50pct/
    └── YOUR_DATASET_100pct/

~/rlwrld_isaac/gr00t/
├── configs/
│   ├── groot_YOUR_DATASET_10pct.yaml   ← Phase 2 output
│   ├── groot_YOUR_DATASET_50pct.yaml
│   └── groot_YOUR_DATASET_100pct.yaml
│
└── automating_groot/
    ├── run_pipeline.py
    ├── phase1_split.py
    ├── phase2_configs.py
    ├── phase3_train.py
    └── sbatch_scripts/
        ├── train_YOUR_DATASET_10pct.sh  ← Phase 3 generated
        ├── train_YOUR_DATASET_50pct.sh
        └── train_YOUR_DATASET_100pct.sh
```

---

## Fixed Training Parameters

| Parameter | Value |
|-----------|-------|
| GPUs | 4 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Max steps | 30,000 |
| Save steps | 10,000 |
| Workers | 8 |
| Embodiment tag | new_embodiment |
| Data config | allex_thetwo_ck40_egostereo |

---

## Notes

- Output paths are currently hardcoded to `/rlwrld1/home/yashu/`. Dynamic user paths planned for future release.
- Worker node `worker-node105` is excluded from training jobs due to a known CUDA/flash_attn incompatibility.
- The pipeline uses `rlwrld` partition by default. Use `--partition yashu` if needed.

---

## Author

Yashu — [github.com/ys2064](https://github.com/ys2064)
