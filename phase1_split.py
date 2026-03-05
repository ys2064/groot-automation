"""
Phase 1 - Dataset Splitting
Automatically splits a dataset into 10%, 50%, 100% subsets.
If total episodes < 50, skips splitting and uses 100% only.

Usage (standalone):
    python phase1_split.py \
        --dataset-path /rlwrld-dataset/.../37-Cube_Box_Box-5cmRight-9029bbfd \
        --dataset-name Cube_Box_Box_5cmRight

Usage (from run_pipeline.py):
    from phase1_split import split_dataset
    split_paths = split_dataset(dataset_path, dataset_name)
"""

import json
import subprocess
import argparse
from pathlib import Path


# ── Fixed paths (do not change) ──────────────────────────────────────
SAMPLER_SCRIPT  = "/rlwrld1/home/yashu/sample_lerobot_dataset_updated.py"
OUTPUT_BASE     = "/rlwrld1/home/yashu/dataset"
SPLITS          = [10, 50, 100]
MIN_EPS_TO_SPLIT = 50   # if total episodes < this, skip splitting


def get_total_episodes(dataset_path: str) -> int:
    """Read total episode count from dataset's info.json"""
    info_path = Path(dataset_path) / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(
            f"Cannot find info.json at: {info_path}\n"
            f"Make sure the dataset path is correct."
        )

    with open(info_path) as f:
        info = json.load(f)

    total = info["total_episodes"]
    print(f"[Phase 1] Found total_episodes = {total}")
    return total


def calc_end_index(total: int, pct: int) -> int:
    """Calculate the end index for a given percentage split."""
    return int(total * pct / 100) - 1


def split_dataset(dataset_path: str, dataset_name: str) -> dict:
    """
    Split a dataset into 10%, 50%, 100% subsets automatically.
    If total episodes < 50, skips splitting and uses full dataset (100%) only.

    Args:
        dataset_path : Full path to the input dataset
        dataset_name : Short name for output folders

    Returns:
        dict e.g. {10: '/path/-10pct', 50: '/path/-50pct', 100: '/path/-100pct'}
             OR   {100: '/path/-100pct'}  if episodes < 50
    """
    print(f"\n{'='*60}")
    print(f"[Phase 1] Starting dataset split")
    print(f"[Phase 1] Input  : {dataset_path}")
    print(f"[Phase 1] Name   : {dataset_name}")
    print(f"{'='*60}\n")

    total        = get_total_episodes(dataset_path)
    output_paths = {}

    # ── Check if dataset is too small to split ────────────────────────
    if total < MIN_EPS_TO_SPLIT:
        print(f"[Phase 1] ⚠️  Only {total} episodes found (< {MIN_EPS_TO_SPLIT})")
        print(f"[Phase 1] ⏭️  Skipping 10%/50% splits — using full dataset (100%) only\n")
        splits_to_run = [100]
    else:
        print(f"[Phase 1] ✅ {total} episodes found — running full 10/50/100% splits\n")
        splits_to_run = SPLITS
    # ─────────────────────────────────────────────────────────────────

    for pct in splits_to_run:
        end_idx    = calc_end_index(total, pct)
        output_dir = f"{OUTPUT_BASE}/{dataset_name}-{pct}pct"

        print(f"[Phase 1] --- {pct}% split ---")
        print(f"[Phase 1]   episodes   : 0 to {end_idx} ({end_idx+1} episodes)")
        print(f"[Phase 1]   output_dir : {output_dir}")

        cmd = [
            "python3", SAMPLER_SCRIPT,
            dataset_path,
            output_dir,
            "0",
            str(end_idx)
        ]

        # Skip if folder already exists
        if Path(output_dir).exists():
            print(f"[Phase 1] ⏭️  Already exists, skipping → {output_dir}\n")
            output_paths[pct] = output_dir
            continue

        print(f"[Phase 1]   running command...")
        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"[Phase 1] FAILED on {pct}% split!\n"
                f"Command was: {' '.join(cmd)}"
            )

        output_paths[pct] = output_dir
        print(f"[Phase 1] ✅ {pct}% done → {output_dir}\n")

    print(f"{'='*60}")
    print(f"[Phase 1] ✅ All splits complete!")
    for pct, path in output_paths.items():
        print(f"  {pct}%  → {path}")
    print(f"{'='*60}\n")

    return output_paths


# ── Run directly from command line ───────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phase 1: Split dataset")

    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Full path to input dataset e.g. /rlwrld-dataset/.../37-Cube_Box_Box-5cmRight-9029bbfd"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Short name for output folders e.g. Cube_Box_Box_5cmRight"
    )

    args = parser.parse_args()

    result = split_dataset(
        dataset_path = args.dataset_path,
        dataset_name = args.dataset_name
    )

    print("Output paths:", result)
