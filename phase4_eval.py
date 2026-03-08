"""
Phase 4 - Generate SBATCH Scripts and Submit Evaluation Jobs to SLURM

Usage (standalone):
    python phase4_eval.py \
        --dataset-name Cylinder_Tube_Place_test \
        --partition rlwrld

Usage (from run_pipeline.py):
    from phase4_eval import submit_eval_jobs
    eval_job_info = submit_eval_jobs(dataset_name, partition)
"""

import subprocess
import argparse
import re
import yaml
from pathlib import Path


# Fixed paths (do not change)
GROOT_DIR        = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t"
RLWRLD_ISAAC_DIR = "/rlwrld1/home/yashu/rlwrld_isaac"
TRAIN_OUTPUT_BASE= "/rlwrld1/home/yashu/output/train"
EVAL_OUTPUT_BASE = "/rlwrld1/home/yashu/output/eval"
VENV             = "/rlwrld1/home/yashu/rlwrld_isaac/.venv/bin/activate"
SBATCH_DIR       = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot/sbatch_scripts"
EVAL_TASKS_YAML  = "/rlwrld1/home/yashu/rlwrld_isaac/configs/eval_tasks.yaml"
COORDINATOR      = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot/eval_coordinator.py"

# Fixed eval parameters
ISAAC_TASK       = "Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0"
DATA_CONFIG      = "allex_thetwo_ck40_egostereo"
ACTION_TYPE      = "joint_action"
N_EPISODES       = 72
IMAGE_CROP_RATIO = 1.0
IMAGE_HEIGHT     = 224
IMAGE_WIDTH      = 224
DIST_LABELS      = ["0cm", "1cm", "3cm", "5cm", "7cm"]
MODEL_PCTS       = [50, 100]


# Helpers

def dataset_name_to_task_name(dataset_name: str) -> str:
    parts = dataset_name.split("_")
    for i in range(len(parts) - 1, 0, -1):
        if re.match(r'^([0-9]|T[0-9])', parts[i]):
            base   = "_".join(parts[:i])
            suffix = "_".join(parts[i:])
            return f"task-{base}-{suffix}"
    return f"task-{dataset_name.replace('_', '-')}"


def get_instruction(task_name: str) -> str:
    with open(EVAL_TASKS_YAML, "r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("tasks", {})
    if task_name not in tasks:
        raise ValueError(
            f"[Phase 4] Task '{task_name}' not found in eval_tasks.yaml!\n"
            f"Available tasks:\n" + "\n".join(f"  - {t}" for t in tasks.keys())
        )
    return tasks[task_name]["instruction"]


def get_checkpoint_path(dataset_name: str, pct: int) -> str:
    ckpt = f"{TRAIN_OUTPUT_BASE}/{dataset_name}_{pct}pct/checkpoint-30000"
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"[Phase 4] Checkpoint not found: {ckpt}\n"
            f"  -> Has Phase 3 training finished for {dataset_name} {pct}%?"
        )
    return ckpt


# SBATCH Generator

def generate_eval_sbatch(
    dataset_name: str,
    task_name:    str,
    instruction:  str,
    pct:          int,
    checkpoint:   str,
    partition:    str = "rlwrld"
) -> str:

    job_name   = f"eval_{dataset_name}_{pct}pct"
    output_dir = f"{EVAL_OUTPUT_BASE}/{dataset_name}/groot{pct}"
    n_dists    = len(DIST_LABELS)

    Path(SBATCH_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{RLWRLD_ISAAC_DIR}/slurm-logs").mkdir(parents=True, exist_ok=True)

    sbatch_path      = f"{SBATCH_DIR}/eval_{job_name}.sh"
    dist_labels_bash = " ".join(f'"{d}"' for d in DIST_LABELS)

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={RLWRLD_ISAAC_DIR}/slurm-logs/%A_%a-{job_name}.out
#SBATCH --error={RLWRLD_ISAAC_DIR}/slurm-logs/%A_%a-{job_name}.err
#SBATCH --partition={partition}
#SBATCH --qos=highprio
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-{n_dists - 1}%{n_dists}

set -e
set -x

# ------------------------------------------------------------------
# 0. Setup & Environment
# ------------------------------------------------------------------
sleep $((RANDOM % 30))

PROJECT_ROOT="{RLWRLD_ISAAC_DIR}"
GR00T_ROOT="{GROOT_DIR}"

mkdir -p ${{PROJECT_ROOT}}/slurm-logs
cd ${{PROJECT_ROOT}}

export PYTHONPATH=${{GR00T_ROOT}}:$PYTHONPATH
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_ATTENTION_TYPES=eager
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

source ~/.bashrc
source {VENV}

# ------------------------------------------------------------------
# 1. Distance Array
# ------------------------------------------------------------------
DIST_LABELS=({dist_labels_bash})
CURRENT_DIST=${{DIST_LABELS[$SLURM_ARRAY_TASK_ID]}}

# ------------------------------------------------------------------
# 2. Output Directory
# ------------------------------------------------------------------
OUTPUT_DIR="{output_dir}/${{CURRENT_DIST}}"
mkdir -p "${{OUTPUT_DIR}}"

# ------------------------------------------------------------------
# 3. Port & Cleanup
# ------------------------------------------------------------------
find_available_port() {{
  while true; do
    port=$((RANDOM % 64511 + 1024))
    if ! ss -tuln | grep -q ":$port "; then
      echo $port
      return
    fi
  done
}}
ALLEX_ENV_PORT=$(find_available_port)

cleanup() {{
  echo "Cleaning up port ${{ALLEX_ENV_PORT}}..."
  kill -9 ${{allex_env_pid}} 2>/dev/null || true
  fuser -k ${{ALLEX_ENV_PORT}}/tcp 2>/dev/null || true
  sleep 2
}}
trap cleanup EXIT

# ------------------------------------------------------------------
# 4. Launch Isaac Sim Server
# ------------------------------------------------------------------
python ${{PROJECT_ROOT}}/scripts/environments/server_v2.py \\
  --task {ISAAC_TASK} \\
  --task_name "{task_name}" \\
  --port "${{ALLEX_ENV_PORT}}" \\
  --device cpu \\
  --image_crop_ratio {IMAGE_CROP_RATIO} \\
  --image_resize_height {IMAGE_HEIGHT} \\
  --image_resize_width {IMAGE_WIDTH} \\
  --eval_set "${{CURRENT_DIST}}" \\
  > "${{PROJECT_ROOT}}/slurm-logs/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_server.log" 2>&1 &
allex_env_pid=$!

# Wait for server with timeout
echo "Waiting for Isaac Sim Server on port ${{ALLEX_ENV_PORT}}..."
set +e
timeout 300s bash -c "until ss -tuln | grep -q \\":${{ALLEX_ENV_PORT}} \\"; do sleep 5; done"
SERVER_WAIT_CODE=$?
set -e

if [ $SERVER_WAIT_CODE -ne 0 ]; then
  echo "Isaac Sim Server did NOT start within 300s — aborting"
  python3 -c "
import sys
sys.path.insert(0, '{GROOT_DIR}/automating_groot')
from notify import notify_error
notify_error(
    'Phase 4 - Isaac Sim Server Timeout',
    '{dataset_name}_{pct}pct ${{CURRENT_DIST}}',
    'Server did not start on port ${{ALLEX_ENV_PORT}} within 300s. Check server log.'
)
"
  exit 1
fi

sleep 10

# ------------------------------------------------------------------
# 5. Launch Evaluation
# ------------------------------------------------------------------
echo "Starting Eval: groot{pct} at ${{CURRENT_DIST}}"

set +e
python ${{GR00T_ROOT}}/scripts/eval_allex.py \\
  --model-path "{checkpoint}" \\
  --server-port "${{ALLEX_ENV_PORT}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --instruction "{instruction}" \\
  --n-episodes {N_EPISODES} \\
  --save-data \\
  --data_config {DATA_CONFIG} \\
  --action_type {ACTION_TYPE}

EVAL_EXIT_CODE=$?
set -e

# ------------------------------------------------------------------
# 6. Verify MP4s and notify complete or error
# ------------------------------------------------------------------
python3 -c "
import sys, os, glob
sys.path.insert(0, '{GROOT_DIR}/automating_groot')
from notify import notify_eval_complete, notify_error

output_dir = '${{OUTPUT_DIR}}'
dataset    = '{dataset_name}'
pct        = {pct}
dist       = '${{CURRENT_DIST}}'
eval_exit  = $EVAL_EXIT_CODE
expected   = {N_EPISODES}

if eval_exit != 0:
    notify_error(
        'Phase 4 - Evaluation Crashed',
        f'{{dataset}}_{{pct}}pct {{dist}}',
        f'eval_allex.py exited with code {{eval_exit}}'
    )
    sys.exit(1)

mp4_files = glob.glob(os.path.join(output_dir, '**', '*.mp4'), recursive=True)
mp4_count = len(mp4_files)

if mp4_count == expected:
    notify_eval_complete(dataset, pct, dist, output_dir, mp4_count)
else:
    notify_error(
        'Phase 4 - Missing Videos',
        f'{{dataset}}_{{pct}}pct {{dist}}',
        f'Expected {{expected}} MP4s but only found {{mp4_count}} in {{output_dir}}'
    )
    sys.exit(1)
"
"""

    Path(sbatch_path).write_text(content)
    print(f"[Phase 4] Generated: eval_{job_name}.sh")
    return sbatch_path


# Submit

def submit_job(sbatch_path: str) -> str:
    result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[Phase 4] sbatch FAILED!\nError: {result.stderr}"
        )
    job_id = result.stdout.strip().split()[-1]
    print(f"[Phase 4] Submitted -> Job ID: {job_id}")
    return job_id


# Launch coordinator in background

def launch_coordinator(dataset_name: str, job_ids: list, eval_job_info: dict):
    """
    Launch eval_coordinator.py as a background process on the login node.
    Watches SLURM and fires ONE notification when all tasks are Running.
    """
    total_tasks = len(job_ids) * len(DIST_LABELS)
    job_ids_str = ",".join(job_ids)
    job_map_str = ",".join([f"{pct}:{info['job_id']}" for pct, info in eval_job_info.items()])

    cmd = [
        "python3", COORDINATOR,
        "--dataset-name",  dataset_name,
        "--job-ids",       job_ids_str,
        "--job-map",       job_map_str,
        "--total-tasks",   str(total_tasks),
        "--poll-interval", "30"
    ]

    process = subprocess.Popen(
        cmd,
        stdout            = open(f"/tmp/coordinator_{dataset_name}.log", "w"),
        stderr            = subprocess.STDOUT,
        start_new_session = True
    )

    print(f"[Phase 4] Coordinator launched (PID {process.pid})")
    print(f"[Phase 4] Watching {total_tasks} tasks across jobs: {job_ids_str}")
    print(f"[Phase 4] Coordinator log: /tmp/coordinator_{dataset_name}.log")


# Main function

def submit_eval_jobs(
    dataset_name: str,
    partition:    str = "rlwrld",
    task_name_override: str = None
) -> dict:

    print(f"\n{'='*60}")
    print(f"[Phase 4] Submitting evaluation jobs")
    print(f"[Phase 4] Dataset  : {dataset_name}")
    print(f"[Phase 4] Models   : {MODEL_PCTS}")
    print(f"[Phase 4] Distances: {DIST_LABELS}")
    print(f"{'='*60}\n")

    task_name   = task_name_override if task_name_override else dataset_name_to_task_name(dataset_name)
    instruction = get_instruction(task_name)

    print(f"[Phase 4] Task name  : {task_name}")
    print(f"[Phase 4] Instruction: {instruction}\n")

    eval_job_info      = {}
    submitted_job_ids  = []

    for pct in MODEL_PCTS:
        checkpoint  = get_checkpoint_path(dataset_name, pct)
        sbatch_path = generate_eval_sbatch(
            dataset_name = dataset_name,
            task_name    = task_name,
            instruction  = instruction,
            pct          = pct,
            checkpoint   = checkpoint,
            partition    = partition
        )
        job_id = submit_job(sbatch_path)
        submitted_job_ids.append(job_id)
        eval_job_info[pct] = {
            "job_id"      : job_id,
            "sbatch_path" : sbatch_path,
            "checkpoint"  : checkpoint,
            "task_name"   : task_name,
            "instruction" : instruction,
        }

    print(f"\n{'='*60}")
    print(f"[Phase 4] All eval jobs submitted!")
    for pct, info in eval_job_info.items():
        print(f"  groot{pct}  -> Job ID: {info['job_id']}  ({len(DIST_LABELS)} distances as array)")
    print(f"{'='*60}\n")

    # Launch coordinator to watch for all tasks Running
    launch_coordinator(dataset_name, submitted_job_ids, eval_job_info)

    return eval_job_info


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Submit evaluation jobs")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--partition",    default="rlwrld")
    parser.add_argument("--task-name",    default=None)
    args = parser.parse_args()

    submit_eval_jobs(
        dataset_name       = args.dataset_name,
        partition          = args.partition,
        task_name_override = args.task_name
    )