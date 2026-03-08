"""
eval_coordinator.py - Watches SLURM and fires ONE Slack notification
when ALL eval array tasks are Running (R).

Called automatically by phase4_eval.py after job submission.
Runs in background on the login node.

Usage (internal):
    python eval_coordinator.py \
        --dataset-name Cylinder_Tube_Place_test \
        --job-ids 12345,12346 \
        --total-tasks 10
"""

import subprocess
import argparse
import time
import sys
from pathlib import Path

# Make notify importable
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from notify import notify_eval_all_started, notify_error


def get_running_count(job_ids: list) -> tuple:
    """
    Query SLURM for all tasks across given job IDs.
    Returns (running_count, total_count, failed_count)
    """
    running = 0
    total   = 0
    failed  = 0

    for job_id in job_ids:
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--job", job_id,
                    "--noheader",
                    "--format=%T"   # %T = state: PENDING, RUNNING, FAILED etc
                ],
                capture_output = True,
                text           = True
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            for state in lines:
                total += 1
                if state == "RUNNING":
                    running += 1
                elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    failed += 1

        except Exception as e:
            print(f"[coordinator] squeue error for job {job_id}: {e}", flush=True)

    return running, total, failed


def watch_until_all_running(
    dataset_name: str,
    job_ids:      list,
    total_tasks:  int,
    poll_interval: int = 30
):
    """
    Poll SLURM every poll_interval seconds until all tasks are Running.
    Then fire ONE Slack notification.
    """
    print(f"[coordinator] Started watching {len(job_ids)} jobs, {total_tasks} total tasks", flush=True)
    print(f"[coordinator] Job IDs: {job_ids}", flush=True)
    print(f"[coordinator] Polling every {poll_interval}s...", flush=True)

    while True:
        running, total, failed = get_running_count(job_ids)

        print(f"[coordinator] Status: {running} Running / {total} Active / {failed} Failed  (target: {total_tasks})", flush=True)

        # ── Check for failures ────────────────────────────────────────
        if failed > 0:
            notify_error(
                "Phase 4 - Coordinator",
                dataset_name,
                f"{failed} eval task(s) failed/cancelled before reaching Running state"
            )
            print(f"[coordinator] ❌ {failed} tasks failed — exiting", flush=True)
            sys.exit(1)

        # ── All tasks running ─────────────────────────────────────────
        if running == total_tasks:
            print(f"[coordinator] ✅ All {total_tasks} tasks are Running — sending notification", flush=True)
            notify_eval_all_started(dataset_name, job_ids, total_tasks)
            break

        # ── All tasks disappeared from queue (jobs completed very fast) ─
        if total == 0:
            print(f"[coordinator] All tasks already finished — exiting without started notification", flush=True)
            break

        time.sleep(poll_interval)

    print(f"[coordinator] Done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval coordinator — fires single started notification")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--job-ids",      required=True, help="Comma separated job IDs e.g. 12345,12346")
    parser.add_argument("--total-tasks",  required=True, type=int)
    parser.add_argument("--poll-interval", default=30,   type=int)
    args = parser.parse_args()

    job_ids = [j.strip() for j in args.job_ids.split(",")]

    watch_until_all_running(
        dataset_name  = args.dataset_name,
        job_ids       = job_ids,
        total_tasks   = args.total_tasks,
        poll_interval = args.poll_interval
    )
