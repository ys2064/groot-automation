#!/bin/bash
#SBATCH --job-name=Cube_Stack_3cmRight_100pct
#SBATCH --output=/rlwrld1/home/yashu/rlwrld_isaac/gr00t/log/%j-Cube_Stack_3cmRight_100pct.out
#SBATCH --error=/rlwrld1/home/yashu/rlwrld_isaac/gr00t/log/%j-Cube_Stack_3cmRight_100pct.err
#SBATCH --partition=rlwrld
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node105

# Standard Environment Setup
source /rlwrld1/home/yashu/rlwrld_isaac/.venv/bin/activate

# Hardware & Env Setup
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Define Output
OUT_DIR="/rlwrld1/home/yashu/output/train/Cube_Stack_3cmRight_100pct"
mkdir -p "$OUT_DIR"

# Launch Training
cd /rlwrld1/home/yashu/rlwrld_isaac/gr00t
python scripts/gr00t_finetune.py   --num-gpus 4   --batch-size 32   --learning_rate 1e-4   --tune-visual   --output-dir "$OUT_DIR"   --data-config /rlwrld1/home/yashu/rlwrld_isaac/gr00t/configs/groot_Cube_Stack_3cmRight_100pct.yaml   --max-steps 30000   --save-steps 10000   --dataloader-num-workers 8
