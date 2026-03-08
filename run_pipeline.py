"""
run_pipeline.py - Master Script
Runs Phase 1, 2, 3 and launches Phase 4 watcher in background.

Usage:
    python run_pipeline.py \
        --dataset-path /rlwrld-dataset/.../3-Cube_Stack-3cmRight-045004d9 \
        --dataset-name Cube_Stack-3cmRight \
        --partition rlwrld
"""

import argparse
import sys
import subprocess
from pathlib import Path

# ── Make imports work from ANY directory ──────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
# ──────────────────────────────────────────────────────────────────────

from phase1_split   import split_dataset
from phase2_configs import generate_yaml_configs
from phase3_train   import submit_training_jobs

from notify import (
    notify_pipeline_start,
    notify_phase1_done,
    notify_phase2_done,
    notify_phase3_done,
    notify_training_in_progress,
    notify_error,
)

WATCHER_SCRIPT = str(PIPELINE_DIR / "phase4_watcher.py")


def launch_phase4_watcher(dataset_name: str, partition: str, task_name: str = None):
    """
    Launch phase4_watcher.py as a fully detached background process.
    Safe to close terminal after this.
    """
    cmd = [
        "python3", WATCHER_SCRIPT,
        "--dataset-name",  dataset_name,
        "--partition",     partition,
        "--poll-interval", "120",
    ]
    if task_name:
        cmd += ["--task-name", task_name]

    log_path = f"/tmp/phase4_watcher_{dataset_name}.log"

    process = subprocess.Popen(
        cmd,
        stdout            = open(log_path, "w"),
        stderr            = subprocess.STDOUT,
        start_new_session = True   # fully detached from terminal
    )

    print(f"\n[Pipeline] Phase 4 watcher launched in background (PID {process.pid})")
    print(f"[Pipeline] Will auto-submit eval once checkpoints are ready")
    print(f"[Pipeline] Watcher log: {log_path}")
    print(f"[Pipeline] Safe to close terminal — Slack will notify you!\n")


def run_pipeline(
    dataset_path:       str,
    dataset_name:       str,
    partition:          str = "rlwrld",
    task_name_override: str = None,
) -> dict:

    print(f"\n###########################################################")
    print(f"# GROOT AUTOMATION PIPELINE")
    print(f"# Dataset  : {dataset_name}")
    print(f"# Path     : {dataset_path}")
    print(f"# Partition: {partition}")
    print(f"###########################################################\n")

    notify_pipeline_start(dataset_name, dataset_path, partition)

    # ── Phase 1 - Split Dataset ────────────────────────────────────────
    try:
        print(">>> PHASE 1: Splitting dataset...")
        split_paths = split_dataset(
            dataset_path = dataset_path,
            dataset_name = dataset_name
        )
        notify_phase1_done(dataset_name, split_paths)
    except Exception as e:
        notify_error("Phase 1 - Split Dataset", dataset_name, str(e))
        raise

    # ── Phase 2 - Generate YAML Configs ───────────────────────────────
    try:
        print(">>> PHASE 2: Generating YAML configs...")
        yaml_paths = generate_yaml_configs(
            dataset_name = dataset_name,
            split_paths  = split_paths
        )
        notify_phase2_done(dataset_name, yaml_paths)
    except Exception as e:
        notify_error("Phase 2 - YAML Configs", dataset_name, str(e))
        raise

    # ── Phase 3 - Submit Training Jobs ────────────────────────────────
    try:
        print(">>> PHASE 3: Submitting training jobs to SLURM...")
        job_info = submit_training_jobs(
            dataset_name = dataset_name,
            yaml_paths   = yaml_paths,
            partition    = partition
        )
        notify_phase3_done(dataset_name, job_info)
    except Exception as e:
        notify_error("Phase 3 - SLURM Submit", dataset_name, str(e))
        raise

    notify_training_in_progress(dataset_name, job_info)

    # ── Launch Phase 4 Watcher in Background ──────────────────────────
    launch_phase4_watcher(dataset_name, partition, task_name_override)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n###########################################################")
    print(f"# PIPELINE SUBMITTED!")
    print(f"# Training jobs running on SLURM:")
    for pct, info in job_info.items():
        print(f"#   {pct}%  -> Job ID: {info['job_id']}")
    print(f"#")
    print(f"# Phase 4 watcher running in background")
    print(f"# Will auto-submit eval after checkpoint-30000 is ready")
    print(f"# You will be notified on Slack at every step!")
    print(f"###########################################################\n")

    return {"training": job_info}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GR00T Automation Pipeline - Runs Phase 1, 2, 3 and 4"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--partition",    default="rlwrld")
    parser.add_argument("--task-name",    default=None)

    args = parser.parse_args()

    run_pipeline(
        dataset_path       = args.dataset_path,
        dataset_name       = args.dataset_name,
        partition          = args.partition,
        task_name_override = args.task_name,
    )

    sys.exit(0)

#Now the flow is:

#python run_pipeline.py ...
   # → Phase 1 ✅
   # → Phase 2 ✅
    #→ Phase 3 ✅ (SLURM jobs submitted)
   # → phase4_watcher.py launched in background (detached)
   # → Terminal returns immediately ✅
   # → You can close laptop ✅

#[background, hours later]
    #→ phase4_watcher detects checkpoint-30000
    #→ submits Phase 4 eval jobs
    #→ coordinator watches eval jobs
    #→ Slack: ✅ Phase 4 Complete