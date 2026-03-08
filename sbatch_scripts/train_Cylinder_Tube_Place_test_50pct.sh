#!/bin/bash
#SBATCH --job-name=Cylinder_Tube_Place_test_50pct
#SBATCH --output=/rlwrld1/home/yashu/rlwrld_isaac/gr00t/log/%j-Cylinder_Tube_Place_test_50pct.out
#SBATCH --error=/rlwrld1/home/yashu/rlwrld_isaac/gr00t/log/%j-Cylinder_Tube_Place_test_50pct.err
#SBATCH --partition=rlwrld
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --requeue

# Standard Environment Setup
source /rlwrld1/home/yashu/rlwrld_isaac/.venv/bin/activate

# Hardware & Env Setup
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Define Output
OUT_DIR="/rlwrld1/home/yashu/output/train/Cylinder_Tube_Place_test_50pct"
mkdir -p "$OUT_DIR"

# ── Checkpoint Watcher (runs in background) ───────────────────────────
python3 /rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot/sbatch_scripts/watcher_Cylinder_Tube_Place_test_50pct.py &
WATCHER_PID=$!
echo "[slurm] Checkpoint watcher started with PID $WATCHER_PID"

# ── Launch Training ───────────────────────────────────────────────────
cd /rlwrld1/home/yashu/rlwrld_isaac/gr00t
python scripts/gr00t_finetune.py \
  --num-gpus 4 \
  --batch-size 32 \
  --learning_rate 1e-4 \
  --tune-visual \
  --output-dir "$OUT_DIR" \
  --data-config /rlwrld1/home/yashu/rlwrld_isaac/gr00t/configs/groot_Cylinder_Tube_Place_test_50pct.yaml \
  --max-steps 30000 \
  --save-steps 10000 \
  --dataloader-num-workers 8

# ── Training Done ─────────────────────────────────────────────────────
touch "$OUT_DIR/.training_done"   # signal watcher to stop
wait $WATCHER_PID                 # wait for watcher to finish

# ── Notify Slack: Training Complete ──────────────────────────────────
python3 -c "
import sys
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_training_complete
notify_training_complete('Cylinder_Tube_Place_test', 50, '$OUT_DIR')
"
