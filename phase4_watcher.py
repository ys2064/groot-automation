"""
phase4_watcher.py - Waits for training checkpoints then submits Phase 4.
Runs as a detached background process — safe to close terminal.

Called automatically by run_pipeline.py after Phase 3 submission.
Log: /tmp/phase4_watcher_{dataset_name}.log
"""

import sys
import time
import argparse
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from phase4_eval import submit_eval_jobs, MODEL_PCTS, TRAIN_OUTPUT_BASE
from notify import notify_error

CHECKPOINT_STEP          = 30000
CHECKPOINT_POLL_INTERVAL = 120  # 2 minutes


def wait_for_checkpoints(dataset_name: str, pcts: list, poll_interval: int):
    print(f"[phase4_watcher] Waiting for checkpoints — polling every {poll_interval}s", flush=True)

    while True:
        all_ready = True
        for pct in pcts:
            ckpt = Path(f"{TRAIN_OUTPUT_BASE}/{dataset_name}_{pct}pct/checkpoint-{CHECKPOINT_STEP}")
            if ckpt.exists():
                print(f"[phase4_watcher] ✅ groot{pct} checkpoint ready", flush=True)
            else:
                print(f"[phase4_watcher] ⏳ groot{pct} not ready yet", flush=True)
                all_ready = False

        if all_ready:
            print(f"[phase4_watcher] All checkpoints found — submitting Phase 4!", flush=True)
            break

        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name",  required=True)
    parser.add_argument("--partition",     default="rlwrld")
    parser.add_argument("--task-name",     default=None)
    parser.add_argument("--poll-interval", default=CHECKPOINT_POLL_INTERVAL, type=int)
    args = parser.parse_args()

    try:
        wait_for_checkpoints(args.dataset_name, MODEL_PCTS, args.poll_interval)
        submit_eval_jobs(
            dataset_name       = args.dataset_name,
            partition          = args.partition,
            task_name_override = args.task_name,
        )
        print(f"[phase4_watcher] Phase 4 submitted successfully!", flush=True)
    except Exception as e:
        print(f"[phase4_watcher] ERROR: {e}", flush=True)
        notify_error("Phase 4 - Watcher", args.dataset_name, str(e))
        sys.exit(1)
