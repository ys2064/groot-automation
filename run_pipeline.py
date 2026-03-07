"""
run_pipeline.py - Master Script
Runs Phase 1, 2, 3 and 4 automatically with one command.

Usage (from anywhere):
    python3 /rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot/run_pipeline.py \
        --dataset-path /rlwrld-dataset/.../37-Cube_Box_Box-5cmRight-9029bbfd \
        --dataset-name Cube_Box_Box_5cmRight \
        --partition rlwrld
"""

import argparse
import sys
from pathlib import Path

# ── Make imports work from ANY directory ──────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
# ──────────────────────────────────────────────────────────────────────

# Import all phases
from phase1_split   import split_dataset
from phase2_configs import generate_yaml_configs
from phase3_train   import submit_training_jobs
from phase4_eval    import submit_eval_jobs          # ← NEW

# Import notifications
from notify import (
    notify_pipeline_start,
    notify_phase1_done,
    notify_phase2_done,
    notify_phase3_done,
    notify_training_in_progress,
    notify_error,
)


def run_pipeline(
    dataset_path: str,
    dataset_name: str,
    partition:    str = "rlwrld"
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

    # ── Phase 4 - Submit Eval Jobs ────────────────────────────────────  ← NEW
    try:                                                                 # ← NEW
        print(">>> PHASE 4: Submitting evaluation jobs to SLURM...")    # ← NEW
        eval_job_info = submit_eval_jobs(                               # ← NEW
            dataset_name = dataset_name,                                # ← NEW
            partition    = partition                                     # ← NEW
        )                                                               # ← NEW
    except Exception as e:                                              # ← NEW
        notify_error("Phase 4 - Eval Submit", dataset_name, str(e))    # ← NEW
        raise                                                           # ← NEW

    # ── Summary ───────────────────────────────────────���───────────────
    print(f"\n###########################################################")
    print(f"# PIPELINE COMPLETE!")
    print(f"# Training jobs:")
    for pct, info in job_info.items():
        print(f"#   {pct}%  -> Job ID: {info['job_id']}")
    print(f"#")
    print(f"# Eval jobs:")                                              # ← NEW
    for pct, info in eval_job_info.items():                            # ← NEW
        print(f"#   groot{pct}  -> Job ID: {info['job_id']}")         # ← NEW
    print(f"###########################################################\n")

    return {"training": job_info, "eval": eval_job_info}               # ← NEW


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GR00T Automation Pipeline - Runs Phase 1, 2, 3 and 4"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--partition", default="rlwrld")

    args = parser.parse_args()

    run_pipeline(
        dataset_path = args.dataset_path,
        dataset_name = args.dataset_name,
        partition    = args.partition
    )

    sys.exit(0)
