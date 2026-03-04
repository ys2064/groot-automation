"""
Phase 2 - Generate YAML Config Files
Auto-generates one YAML training config for each dataset split.

Usage (standalone):
    python phase2_configs.py \
        --dataset-name Cube_Box_Box_5cmRight

Usage (from run_pipeline.py):
    from phase2_configs import generate_yaml_configs
    yaml_paths = generate_yaml_configs(dataset_name, split_paths)
"""

import argparse
from pathlib import Path


# Fixed values (do not change)
CONFIGS_DIR    = "/rlwrld1/home/yashu/rlwrld_isaac/gr00t/configs"
EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG    = "allex_thetwo_ck40_egostereo"
WEIGHT         = 1.0
OUTPUT_BASE    = "/rlwrld1/home/yashu/dataset"


def generate_yaml_configs(dataset_name: str, split_paths: dict) -> dict:
    """
    Generate YAML config files for each dataset split.

    Args:
        dataset_name : Short name e.g. Cube_Box_Box_5cmRight
        split_paths  : Output from Phase 1
                       {10: /path/-10pct, 50: /path/-50pct, 100: /path/-100pct}

    Returns:
        dict {10: /path/to/yaml, 50: /path/to/yaml, 100: /path/to/yaml}
    """
    print(f"\n{'='*60}")
    print(f"[Phase 2] Generating YAML configs for: {dataset_name}")
    print(f"{'='*60}\n")

    configs_dir = Path(CONFIGS_DIR)
    configs_dir.mkdir(parents=True, exist_ok=True)

    yaml_paths = {}

    for pct, dataset_path in split_paths.items():

        yaml_filename = f"groot_{dataset_name}_{pct}pct.yaml"
        yaml_path     = configs_dir / yaml_filename

        # Skip if already exists
        if yaml_path.exists():
            print(f"[Phase 2] Already exists, skipping -> {yaml_filename}")
            yaml_paths[pct] = str(yaml_path)
            continue

        # Matches your exact YAML format
        content = f"""train:
  datasets:
    - path: {dataset_path}
      embodiment_tag: {EMBODIMENT_TAG}
      data_config: {DATA_CONFIG}
      weight: {WEIGHT}
"""
        yaml_path.write_text(content)
        yaml_paths[pct] = str(yaml_path)

        print(f"[Phase 2] Created: {yaml_filename}")
        print(f"          path   -> {dataset_path}\n")

    print(f"{'='*60}")
    print(f"[Phase 2] All YAML configs ready!")
    for pct, path in yaml_paths.items():
        print(f"  {pct}%  -> {path}")
    print(f"{'='*60}\n")

    return yaml_paths


# Run directly from command line
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phase 2: Generate YAML configs")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Short name e.g. Cube_Box_Box_5cmRight"
    )
    args = parser.parse_args()

    # Rebuild split_paths from Phase 1 output structure
    split_paths = {
        10:  f"{OUTPUT_BASE}/{args.dataset_name}-10pct",
        50:  f"{OUTPUT_BASE}/{args.dataset_name}-50pct",
        100: f"{OUTPUT_BASE}/{args.dataset_name}-100pct",
    }

    yaml_paths = generate_yaml_configs(args.dataset_name, split_paths)

    # Show contents of one file to verify
    print("--- Contents of 10pct yaml ---")
    with open(yaml_paths[10]) as f:
        print(f.read())
