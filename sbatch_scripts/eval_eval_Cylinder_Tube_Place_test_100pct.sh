#!/bin/bash
#SBATCH --job-name=eval_Cylinder_Tube_Place_test_100pct
#SBATCH --output=/rlwrld1/home/yashu/rlwrld_isaac/slurm-logs/%A_%a-eval_Cylinder_Tube_Place_test_100pct.out
#SBATCH --error=/rlwrld1/home/yashu/rlwrld_isaac/slurm-logs/%A_%a-eval_Cylinder_Tube_Place_test_100pct.err
#SBATCH --partition=rlwrld
#SBATCH --requeue
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
OUTPUT_DIR="/rlwrld1/home/yashu/output/eval/Cylinder_Tube_Place_test/groot100/${CURRENT_DIST}"
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
  echo "[cleanup] Triggered — killing Isaac Sim server..."
  kill -9 ${allex_env_pid} 2>/dev/null || true
  fuser -k ${ALLEX_ENV_PORT}/tcp 2>/dev/null || true
  sleep 2
}
trap cleanup EXIT

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

# Wait for server with timeout
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
    'Cylinder_Tube_Place_test_100pct ${CURRENT_DIST}',
    'Server did not start on port ${ALLEX_ENV_PORT} within 300s. Check server log.'
)
"
  exit 1
fi

sleep 10

# ------------------------------------------------------------------
# 5. Launch Evaluation + MP4 Watcher
# ------------------------------------------------------------------
echo "Starting Eval: groot100 at ${CURRENT_DIST}"

set +e

# Launch eval_allex.py in BACKGROUND
python ${GR00T_ROOT}/scripts/eval_allex.py \
  --model-path "/rlwrld1/home/yashu/output/train/Cylinder_Tube_Place_test_100pct/checkpoint-30000" \
  --server-port "${ALLEX_ENV_PORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --instruction "Lift the cylinder with your right hand and place it in the middle of the tube without touching the tube." \
  --n-episodes 72 \
  --save-data \
  --data_config allex_thetwo_ck40_egostereo \
  --action_type joint_action &

EVAL_PID=$!
echo "[eval] eval_allex.py started with PID $EVAL_PID"

# MP4 Watcher runs in FOREGROUND
# Checks every 30s — kills eval once all 72 MP4s are saved
python3 -c "
import os, glob, time, signal, sys

output_dir     = '${OUTPUT_DIR}'
expected       = 72
eval_pid       = $EVAL_PID
check_interval = 30

print(f'[mp4_watcher] Watching for {expected} MP4s in {output_dir}', flush=True)

while True:
    time.sleep(check_interval)

    # Check if eval already finished on its own
    try:
        os.kill(eval_pid, 0)
    except ProcessLookupError:
        print('[mp4_watcher] eval_allex.py already exited on its own', flush=True)
        break

    # Count MP4s
    mp4_files = glob.glob(os.path.join(output_dir, '**', '*.mp4'), recursive=True)
    mp4_count = len(mp4_files)
    print(f'[mp4_watcher] {mp4_count} / {expected} MP4s saved', flush=True)

    if mp4_count >= expected:
        print(f'[mp4_watcher] All {expected} MP4s saved — stopping eval_allex.py (PID {eval_pid})', flush=True)
        try:
            os.kill(eval_pid, signal.SIGTERM)
            time.sleep(5)
            try:
                os.kill(eval_pid, 0)
                os.kill(eval_pid, signal.SIGKILL)
                print('[mp4_watcher] Force killed eval_allex.py', flush=True)
            except ProcessLookupError:
                print('[mp4_watcher] eval_allex.py exited cleanly after SIGTERM', flush=True)
        except ProcessLookupError:
            print('[mp4_watcher] eval_allex.py already gone', flush=True)
        break

print('[mp4_watcher] Done', flush=True)
"

# Wait for eval process to fully exit
wait $EVAL_PID 2>/dev/null || true
set -e

# ------------------------------------------------------------------
# 6. Verify MP4s and notify complete or error
# ------------------------------------------------------------------
python3 -c "
import sys, os, glob
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_eval_complete, notify_error

output_dir = '${OUTPUT_DIR}'
dataset    = 'Cylinder_Tube_Place_test'
pct        = 100
dist       = '${CURRENT_DIST}'
expected   = 72

mp4_files = glob.glob(os.path.join(output_dir, '**', '*.mp4'), recursive=True)
mp4_count = len(mp4_files)

print(f'[verify] {mp4_count} / {expected} MP4s found', flush=True)

if mp4_count >= expected:
    notify_eval_complete(dataset, pct, dist, output_dir, mp4_count)
else:
    notify_error(
        'Phase 4 - Missing Videos',
        f'{dataset}_{pct}pct {dist}',
        f'Expected {expected} MP4s but only found {mp4_count} in {output_dir}'
    )
    sys.exit(1)
"
