"""
Phase 3 - Generate SBATCH Scripts and Submit Training Jobs to SLURM

Usage (standalone):
    python phase3_train.py \
        --dataset-name Cube_Box_Box_5cmRight \
        --partition rlwrld

Usage (from run_pipeline.py):
    from phase3_train import submit_training_jobs
    job_info = submit_training_jobs(dataset_name, yaml_paths, partition)
"""

import subprocess
import argparse
from pathlib import Path


# Fixed paths and values (do not change)
GROOT_DIR   = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t"
OUTPUT_BASE = "/rlwrld1/home/yashu/output/train"
VENV        = "/rlwrld1/home/yashu/rlwrld_isaac/.venv/bin/activate"
SBATCH_DIR  = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot/sbatch_scripts"
CONFIGS_DIR = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t/configs"


def generate_train_sbatch(
    dataset_name: str,
    pct: int,
    yaml_path: str,
    partition: str = "rlwrld"
) -> str:
    job_name   = f"{dataset_name}_{pct}pct"
    output_dir = f"{OUTPUT_BASE}/{dataset_name}_{pct}pct"

    Path(SBATCH_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{GROOT_DIR}/log").mkdir(parents=True, exist_ok=True)

    sbatch_path  = f"{SBATCH_DIR}/train_{job_name}.sh"
    watcher_path = f"{SBATCH_DIR}/watcher_{job_name}.py"

    # Write watcher as a separate Python file
    watcher_script = f"""import os, time, sys
sys.path.insert(0, '{GROOT_DIR}/automating_groot')
from notify import notify_checkpoint_saved, notify_error

output_dir = "{output_dir}"
dataset    = "{dataset_name}"
pct        = {pct}
notified   = set()

print('[watcher] Started, watching:', output_dir, flush=True)

try:
    while True:
        time.sleep(30)
        try:
            entries = os.listdir(output_dir)
        except FileNotFoundError:
            continue
        for entry in sorted(entries):
            if entry.startswith('checkpoint-') and entry not in notified:
                step = int(entry.split('-')[1])
                ckpt_path = os.path.join(output_dir, entry)
                print(f'[watcher] New checkpoint: {{entry}}', flush=True)
                notify_checkpoint_saved(dataset, pct, step, ckpt_path)
                notified.add(entry)
        if os.path.exists(os.path.join(output_dir, '.training_done')):
            print('[watcher] Done flag detected, exiting', flush=True)
            break
except Exception as e:
    notify_error('Phase 3 - Checkpoint Watcher', dataset, str(e))
    sys.exit(1)
"""
    Path(watcher_path).write_text(watcher_script)

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={GROOT_DIR}/log/%j-{job_name}.out
#SBATCH --error={GROOT_DIR}/log/%j-{job_name}.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --requeue

# ── Environment Setup ─────────────────────────────────────────────────
source {VENV}

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ── Define Output ─────────────────────────────────────────────────────
OUT_DIR="{output_dir}"
mkdir -p "$OUT_DIR"

# ── Checkpoint Watcher (runs in background) ───────────────────────────
python3 {watcher_path} &
WATCHER_PID=$!
echo "[slurm] Checkpoint watcher started with PID $WATCHER_PID"

# ── Launch Training ───────────────────────────────────────────────────
cd {GROOT_DIR}

set +e
python scripts/gr00t_finetune.py \\
  --num-gpus 4 \\
  --batch-size 32 \\
  --learning_rate 1e-4 \\
  --tune-visual \\
  --output-dir "$OUT_DIR" \\
  --data-config {yaml_path} \\
  --max-steps 30000 \\
  --save-steps 10000 \\
  --dataloader-num-workers 8

TRAIN_EXIT_CODE=$?
set -e

# ── ✅ Check training exit code ───────────────────────────────────────
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
  echo "[slurm] Training FAILED with exit code $TRAIN_EXIT_CODE"
  python3 -c "
import sys
sys.path.insert(0, '{GROOT_DIR}/automating_groot')
from notify import notify_error
notify_error('Phase 3 - Training Crashed', '{dataset_name}_{pct}pct', 'gr00t_finetune.py exited with code $TRAIN_EXIT_CODE')
"
  # Signal watcher to stop even on failure
  touch "$OUT_DIR/.training_done"
  wait $WATCHER_PID
  exit $TRAIN_EXIT_CODE
fi

# ── Training Done ─────────────────────────────────────────────────────
touch "$OUT_DIR/.training_done"   # signal watcher to stop
wait $WATCHER_PID                 # wait for watcher to finish

# ── Notify Slack: Training Complete ───────────────────────────────────
python3 -c "
import sys
sys.path.insert(0, '{GROOT_DIR}/automating_groot')
from notify import notify_training_complete
notify_training_complete('{dataset_name}', {pct}, '$OUT_DIR')
"
"""
    Path(sbatch_path).write_text(content)
    print(f"[Phase 3] Generated sbatch: train_{job_name}.sh")
    return sbatch_path


def submit_job(sbatch_path: str) -> str:
    result = subprocess.run(
        ["sbatch", sbatch_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[Phase 3] sbatch submission FAILED!\n"
            f"Error: {result.stderr}"
        )

    job_id = result.stdout.strip().split()[-1]
    print(f"[Phase 3] Submitted to SLURM -> Job ID: {job_id}")
    return job_id


def submit_training_jobs(
    dataset_name: str,
    yaml_paths: dict,
    partition: str = "rlwrld"
) -> dict:

    print(f"\n{'='*60}")
    print(f"[Phase 3] Submitting training jobs")
    print(f"[Phase 3] Dataset : {dataset_name}")
    print(f"[Phase 3] Splits  : {list(yaml_paths.keys())}")
    print(f"{'='*60}\n")

    job_info = {}

    for pct, yaml_path in yaml_paths.items():
        sbatch_path = generate_train_sbatch(
            dataset_name = dataset_name,
            pct          = pct,
            yaml_path    = yaml_path,
            partition    = partition
        )
        job_id = submit_job(sbatch_path)
        job_info[pct] = {
            "job_id"      : job_id,
            "sbatch_path" : sbatch_path,
            "yaml_path"   : yaml_path,
        }

    print(f"\n{'='*60}")
    print(f"[Phase 3] ✅ All jobs submitted!")
    for pct, info in job_info.items():
        print(f"  {pct}%  -> Job ID: {info['job_id']}")
    print(f"{'='*60}\n")

    return job_info


# ── Run directly from command line ────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phase 3: Submit training jobs")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--partition", default="rlwrld")
    args = parser.parse_args()