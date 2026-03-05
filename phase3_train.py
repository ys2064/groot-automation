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

    sbatch_path = f"{SBATCH_DIR}/train_{job_name}.sh"

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={GROOT_DIR}/log/%j-{job_name}.out
#SBATCH --error={GROOT_DIR}/log/%j-{job_name}.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --requeue

# Standard Environment Setup
source {VENV}

# Hardware & Env Setup
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Define Output
OUT_DIR="{output_dir}"
mkdir -p "$OUT_DIR"

# Launch Training
cd {GROOT_DIR}
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
    print(f"[Phase 3] Submitting training jobs for: {dataset_name}")
    print(f"[Phase 3] Partition: {partition}")
    print(f"{'='*60}\n")

    job_info = {}

    for pct, yaml_path in yaml_paths.items():
        print(f"[Phase 3] --- {pct}% training job ---")

        sbatch_path = generate_train_sbatch(
            dataset_name, pct, yaml_path, partition
        )
        job_id = submit_job(sbatch_path)

        job_info[pct] = {
            "job_id":      job_id,
            "output_dir":  f"{OUTPUT_BASE}/{dataset_name}_{pct}pct",
            "sbatch_path": sbatch_path,
        }
        print()

    print(f"{'='*60}")
    print(f"[Phase 3] All 3 training jobs submitted!")
    for pct, info in job_info.items():
        print(f"  {pct}%  -> Job ID: {info['job_id']} | output: {info['output_dir']}")
    print(f"{'='*60}\n")

    return job_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phase 3: Submit training jobs")
    parser.add_argument("--dataset-name", required=True, help="Short name e.g. Cube_Box_Box_5cmRight")
    parser.add_argument("--partition", default="rlwrld", help="SLURM partition (default: rlwrld)")
    args = parser.parse_args()

    yaml_paths = {
        10:  f"{CONFIGS_DIR}/groot_{args.dataset_name}_10pct.yaml",
        50:  f"{CONFIGS_DIR}/groot_{args.dataset_name}_50pct.yaml",
        100: f"{CONFIGS_DIR}/groot_{args.dataset_name}_100pct.yaml",
    }

    job_info = submit_training_jobs(
        dataset_name = args.dataset_name,
        yaml_paths   = yaml_paths,
        partition    = args.partition
    )

    print("Job info:", job_info)
