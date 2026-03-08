"""
eval_coordinator.py - Watches SLURM and fires ONE Slack notification
when ALL eval array tasks are Running (R).

Called automatically by phase4_eval.py after job submission.
Runs in background on the login node.
"""

import subprocess
import argparse
import time
import re
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from notify import notify_eval_all_started, notify_error


def get_task_states(job_ids: list) -> tuple:
    """
    Query SLURM for all tasks across given job IDs.
    Correctly handles both:
      - Pending array:  237160_[0-4%5]  (one line = multiple tasks)
      - Running tasks:  237160_0         (one line = one task)

    Returns (running_count, pending_count, failed_count)
    """
    running = 0
    pending = 0
    failed  = 0

    job_ids_str = ",".join(job_ids)

    try:
        result = subprocess.run(
            [
                "squeue",
                "--job",    job_ids_str,
                "--noheader",
                "--format", "%i %T"
            ],
            capture_output = True,
            text           = True
        )

        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            job_id_field = parts[0]
            state        = parts[1]

            # Pending array range e.g. 237160_[0-4%5]
            range_match = re.search(r'\[(\d+)-(\d+)', job_id_field)
            if range_match:
                start   = int(range_match.group(1))
                end     = int(range_match.group(2))
                n_tasks = end - start + 1
                if state == "PENDING":
                    pending += n_tasks
                elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    failed  += n_tasks
            else:
                # Individual task e.g. 237160_0
                if state == "RUNNING":
                    running += 1
                elif state == "PENDING":
                    pending += 1
                elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    failed  += 1

    except Exception as e:
        print(f"[coordinator] squeue error: {e}", flush=True)

    return running, pending, failed


def watch_until_all_running(
    dataset_name:  str,
    job_ids:       list,
    total_tasks:   int,
    eval_job_info: dict,
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
        running, pending, failed = get_task_states(job_ids)
        total_active = running + pending

        print(
            f"[coordinator] Running={running} Pending={pending} "
            f"Failed={failed}  (target: {total_tasks} Running)",
            flush=True
        )

        # Check for failures
        if failed > 0:
            notify_error(
                "Phase 4 - Coordinator",
                dataset_name,
                f"{failed} eval task(s) failed/cancelled before reaching Running state"
            )
            print(f"[coordinator] {failed} tasks failed — exiting", flush=True)
            sys.exit(1)

        # All tasks running
        if running >= total_tasks:
            print(f"[coordinator] All {total_tasks} tasks are Running — sending notification", flush=True)
            notify_eval_all_started(dataset_name, eval_job_info)
            break

        # All tasks disappeared from queue (already finished very fast)
        if total_active == 0:
            print(f"[coordinator] All tasks already finished — exiting", flush=True)
            break

        time.sleep(poll_interval)

    print(f"[coordinator] Done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval coordinator — fires single started notification")
    parser.add_argument("--dataset-name",  required=True)
    parser.add_argument("--job-ids",       required=True, help="Comma separated job IDs e.g. 12345,12346")
    parser.add_argument("--job-map",       required=True, help="Comma separated pct:job_id pairs e.g. 50:12345,100:12346")
    parser.add_argument("--total-tasks",   required=True, type=int)
    parser.add_argument("--poll-interval", default=30,    type=int)
    args = parser.parse_args()

    job_ids = [j.strip() for j in args.job_ids.split(",")]

    # Parse job map e.g. "50:237160,100:237161" into dict
    eval_job_info = {}
    for pair in args.job_map.split(","):
        pct_str, job_id = pair.split(":")
        eval_job_info[int(pct_str)] = {"job_id": job_id}

    watch_until_all_running(
        dataset_name  = args.dataset_name,
        job_ids       = job_ids,
        total_tasks   = args.total_tasks,
        eval_job_info = eval_job_info,
        poll_interval = args.poll_interval
    )