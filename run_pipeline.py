"""
run_pipeline.py - Master Script
Runs Phase 1, 2, and 3 automatically with one command.

Usage:
    python run_pipeline.py \
        --dataset-path /rlwrld-dataset/.../37-Cube_Box_Box-5cmRight-9029bbfd \
        --dataset-name Cube_Box_Box_5cmRight \
        --partition rlwrld
"""

import argparse
import sys

# Import all phases
from phase1_split   import split_dataset
from phase2_configs import generate_yaml_configs
from phase3_train   import submit_training_jobs


def run_pipeline(
    dataset_path: str,
    dataset_name: str,
    partition: str = "rlwrld"
) -> dict:
    """
    Run the full training pipeline automatically.

    Args:
        dataset_path : Full path to input dataset
        dataset_name : Short name e.g. Cube_Box_Box_5cmRight
        partition    : SLURM partition (default: rlwrld)

    Returns:
        dict with job_info from Phase 3
    """

    print(f"\n###########################################################")
    print(f"# GROOT AUTOMATION PIPELINE")
    print(f"# Dataset  : {dataset_name}")
    print(f"# Path     : {dataset_path}")
    print(f"# Partition: {partition}")
    print(f"###########################################################\n")

    # Phase 1 - Split Dataset
    print(">>> PHASE 1: Splitting dataset...")
    split_paths = split_dataset(
        dataset_path = dataset_path,
        dataset_name = dataset_name
    )

    # Phase 2 - Generate YAML Configs
    print(">>> PHASE 2: Generating YAML configs...")
    yaml_paths = generate_yaml_configs(
        dataset_name = dataset_name,
        split_paths  = split_paths
    )

    # Phase 3 - Submit Training Jobs
    print(">>> PHASE 3: Submitting training jobs to SLURM...")
    job_info = submit_training_jobs(
        dataset_name = dataset_name,
        yaml_paths   = yaml_paths,
        partition    = partition
    )

    # Summary
    print(f"\n###########################################################")
    print(f"# PIPELINE COMPLETE!")
    print(f"# Training jobs submitted:")
    for pct, info in job_info.items():
        print(f"#   {pct}%  -> Job ID: {info['job_id']}")
    print(f"#")
    print(f"# Monitor jobs with:")
    print(f"#   squeue --me | grep {dataset_name[:8]}")
    print(f"#")
    print(f"# Check logs with:")
    print(f"#   tail -f ~/rlwrld_isaac/gr00t/log/<job_id>-*.out")
    print(f"###########################################################\n")

    return job_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GR00T Automation Pipeline - Runs Phase 1, 2, and 3"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Full path to input dataset"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Short name e.g. Cube_Box_Box_5cmRight"
    )
    parser.add_argument(
        "--partition",
        default="rlwrld",
        help="SLURM partition (default: rlwrld)"
    )

    args = parser.parse_args()

    job_info = run_pipeline(
        dataset_path = args.dataset_path,
        dataset_name = args.dataset_name,
        partition    = args.partition
    )

    sys.exit(0)
