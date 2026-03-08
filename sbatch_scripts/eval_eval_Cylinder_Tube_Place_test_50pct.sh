#!/bin/bash
#SBATCH --job-name=eval_Cylinder_Tube_Place_test_50pct
#SBATCH --output=/rlwrld1/home/yashu/rlwrld_isaac/slurm-logs/%A_%a-eval_Cylinder_Tube_Place_test_50pct.out
#SBATCH --error=/rlwrld1/home/yashu/rlwrld_isaac/slurm-logs/%A_%a-eval_Cylinder_Tube_Place_test_50pct.err
#SBATCH --partition=rlwrld
#SBATCH --qos=highprio
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4%5

set -e
set -x

# ------------------------------------------------------------------
# 0. Setup & Environment
# ------------------------------------------------------------------
sleep $((RANDOM % 30))

PROJECT_ROOT="/rlwrld1/home/yashu/rlwrld_isaac"
GR00T_ROOT="/rlwrld1/home/yashu/rlwrld_isaac/gr00t"

mkdir -p ${PROJECT_ROOT}/slurm-logs
cd ${PROJECT_ROOT}

export PYTHONPATH=${GR00T_ROOT}:$PYTHONPATH
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_ATTENTION_TYPES=eager
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

source ~/.bashrc
source /rlwrld1/home/yashu/rlwrld_isaac/.venv/bin/activate

# ------------------------------------------------------------------
# 1. Distance Array
# ------------------------------------------------------------------
DIST_LABELS=("0cm" "1cm" "3cm" "5cm" "7cm")
CURRENT_DIST=${DIST_LABELS[$SLURM_ARRAY_TASK_ID]}

# ------------------------------------------------------------------
# 2. Output Directory
# ------------------------------------------------------------------
OUTPUT_DIR="/rlwrld1/home/yashu/output/eval/Cylinder_Tube_Place_test/groot50/${CURRENT_DIST}"
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------
# 3. Port & Cleanup
# ------------------------------------------------------------------
find_available_port() {
  while true; do
    port=$((RANDOM % 64511 + 1024))
    if ! ss -tuln | grep -q ":$port "; then
      echo $port
      return
    fi
  done
}
ALLEX_ENV_PORT=$(find_available_port)

cleanup() {
  echo "Cleaning up port ${ALLEX_ENV_PORT}..."
  kill -9 ${allex_env_pid} 2>/dev/null || true
  fuser -k ${ALLEX_ENV_PORT}/tcp 2>/dev/null || true
  sleep 2
}
trap cleanup EXIT

# ------------------------------------------------------------------
# ✅ Notify eval STARTED — fires only when node is Running (R)
# ------------------------------------------------------------------
python3 -c "
import sys
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_eval_started
notify_eval_started('Cylinder_Tube_Place_test', 50, '${CURRENT_DIST}')
"

# ------------------------------------------------------------------
# 4. Launch Isaac Sim Server
# ------------------------------------------------------------------
python ${PROJECT_ROOT}/scripts/environments/server_v2.py \
  --task Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0 \
  --task_name "task-Cylinder_Tube_Place-T15cmC7cmLeft" \
  --port "${ALLEX_ENV_PORT}" \
  --device cpu \
  --image_crop_ratio 1.0 \
  --image_resize_height 224 \
  --image_resize_width 224 \
  --eval_set "${CURRENT_DIST}" \
  > "${PROJECT_ROOT}/slurm-logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_server.log" 2>&1 &
allex_env_pid=$!

# ── ✅ Wait for server with timeout — notify if it never starts ───────
echo "Waiting for Isaac Sim Server on port ${ALLEX_ENV_PORT}..."
set +e
timeout 300s bash -c "until ss -tuln | grep -q \":${ALLEX_ENV_PORT} \"; do sleep 5; done"
SERVER_WAIT_CODE=$?
set -e

if [ $SERVER_WAIT_CODE -ne 0 ]; then
  echo "Isaac Sim Server did NOT start within 300s — aborting"
  python3 -c "
import sys
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_error
notify_error(
    'Phase 4 - Isaac Sim Server Timeout',
    'Cylinder_Tube_Place_test_50pct ${CURRENT_DIST}',
    'Server did not start on port ${ALLEX_ENV_PORT} within 300s. Check server log.'
)
"
  exit 1
fi

sleep 10

# ------------------------------------------------------------------
# 5. Launch Evaluation
# ------------------------------------------------------------------
echo "Starting Eval: groot50 at ${CURRENT_DIST}"

set +e
python ${GR00T_ROOT}/scripts/eval_allex.py \
  --model-path "/rlwrld1/home/yashu/output/train/Cylinder_Tube_Place_test_50pct/checkpoint-30000" \
  --server-port "${ALLEX_ENV_PORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --instruction "Lift the cylinder with your right hand and place it in the middle of the tube without touching the tube." \
  --n-episodes 72 \
  --save-data \
  --data_config allex_thetwo_ck40_egostereo \
  --action_type joint_action

EVAL_EXIT_CODE=$?
set -e

# ------------------------------------------------------------------
# ✅ Verify 72 MP4s exist, then notify complete or error
# ------------------------------------------------------------------
python3 -c "
import sys, os, glob
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_eval_complete, notify_error

output_dir = '${OUTPUT_DIR}'
dataset    = 'Cylinder_Tube_Place_test'
pct        = 50
dist       = '${CURRENT_DIST}'
eval_exit  = $EVAL_EXIT_CODE
expected   = 72

# Step 1: Check eval exited cleanly
if eval_exit != 0:
    notify_error(
        'Phase 4 - Evaluation Crashed',
        f'{dataset}_{pct}pct {dist}',
        f'eval_allex.py exited with code {eval_exit}'
    )
    sys.exit(1)

# Step 2: Count MP4 files in output folder
mp4_files = glob.glob(os.path.join(output_dir, '**', '*.mp4'), recursive=True)
mp4_count = len(mp4_files)

# Step 3: Notify based on count
if mp4_count == expected:
    notify_eval_complete(dataset, pct, dist, output_dir, mp4_count)
else:
    notify_error(
        'Phase 4 - Missing Videos',
        f'{dataset}_{pct}pct {dist}',
        f'Expected {expected} MP4s but only found {mp4_count} in {output_dir}'
    )
    sys.exit(1)
"
