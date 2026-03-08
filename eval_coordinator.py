"""
eval_coordinator.py - Watches SLURM and fires Slack notifications:
  1. ONE "Eval Started" notification when ALL tasks are Running
  2. ONE "Eval Complete" notification when ALL tasks finish, with full summary

Called automatically by phase4_eval.py after job submission.
Runs in background on the login node.
"""

import subprocess
import argparse
import time
import re
import sys
import glob
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from notify import notify_eval_all_started, notify_eval_all_complete, notify_error


def get_task_states(job_ids: list) -> tuple:
    """
    Query SLURM for all tasks across given job IDs.
    Returns (running_count, pending_count, failed_count)
    """
    running = 0
    pending = 0
    failed  = 0

    job_ids_str = ",".join(job_ids)

    try:
        result = subprocess.run(
            ["squeue", "--job", job_ids_str, "--noheader", "--format", "%i %T"],
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
                if state == "RUNNING":
                    running += 1
                elif state == "PENDING":
                    pending += 1
                elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    failed  += 1

    except Exception as e:
        print(f"[coordinator] squeue error: {e}", flush=True)

    return running, pending, failed


def count_mp4s(output_dir: str) -> int:
    mp4_files = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
    return len(mp4_files)


def build_summary(
    dataset_name:     str,
    eval_job_info:    dict,
    dist_labels:      list,
    n_episodes:       int,
    eval_output_base: str
) -> dict:
    """
    Check MP4 counts for all models and distances.
    Returns: { 50: {"0cm": 72, "1cm": 72, ...}, 100: {...} }
    """
    summary = {}
    for pct in eval_job_info.keys():
        summary[pct] = {}
        for dist in dist_labels:
            output_dir       = f"{eval_output_base}/{dataset_name}/groot{pct}/{dist}"
            count            = count_mp4s(output_dir)
            summary[pct][dist] = count
            print(f"[coordinator] groot{pct} {dist}: {count}/{n_episodes} MP4s", flush=True)
    return summary


def watch(
    dataset_name:     str,
    job_ids:          list,
    total_tasks:      int,
    eval_job_info:    dict,
    dist_labels:      list,
    n_episodes:       int,
    eval_output_base: str,
    poll_interval:    int = 30
):
    print(f"[coordinator] Started — watching {len(job_ids)} jobs, {total_tasks} total tasks", flush=True)
    print(f"[coordinator] Job IDs: {job_ids}", flush=True)
    print(f"[coordinator] Polling every {poll_interval}s", flush=True)

    # ── Phase 1: Wait until ALL tasks Running → fire "Eval Started" ──────────
    all_started = False
    while not all_started:
        running, pending, failed = get_task_states(job_ids)
        total_active = running + pending

        print(
            f"[coordinator] [STARTED WATCH] Running={running} Pending={pending} "
            f"Failed={failed}  (target={total_tasks})",
            flush=True
        )

        if failed > 0:
            notify_error(
                "Phase 4 - Coordinator",
                dataset_name,
                f"{failed} eval task(s) failed/cancelled before reaching Running state"
            )
            print(f"[coordinator] {failed} tasks failed — exiting", flush=True)
            sys.exit(1)

        if running >= total_tasks:
            print(f"[coordinator] All {total_tasks} tasks Running — sending start notification", flush=True)
            notify_eval_all_started(dataset_name, eval_job_info)
            all_started = True
            break

        if total_active == 0:
            print(f"[coordinator] All tasks finished very fast — skipping start notification", flush=True)
            all_started = True
            break

        time.sleep(poll_interval)

    # ── Phase 2: Wait until ALL tasks finish ─────────────────────────────────
    print(f"[coordinator] Now waiting for all tasks to finish...", flush=True)

    while True:
        running, pending, failed = get_task_states(job_ids)
        total_active = running + pending

        print(
            f"[coordinator] [FINISH WATCH] Running={running} Pending={pending} Failed={failed}",
            flush=True
        )

        if total_active == 0:
            print(f"[coordinator] All tasks finished — building MP4 summary...", flush=True)
            break

        time.sleep(poll_interval)

    # ── Phase 3: Build summary and send ONE combined notification ─────────────
    summary = build_summary(dataset_name, eval_job_info, dist_labels, n_episodes, eval_output_base)
    notify_eval_all_complete(dataset_name, summary, dist_labels, n_episodes)

    print(f"[coordinator] Done ✅", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name",     required=True)
    parser.add_argument("--job-ids",          required=True, help="Comma separated job IDs e.g. 12345,12346")
    parser.add_argument("--job-map",          required=True, help="Comma separated pct:job_id pairs e.g. 50:12345,100:12346")
    parser.add_argument("--total-tasks",      required=True, type=int)
    parser.add_argument("--poll-interval",    default=30,    type=int)
    parser.add_argument("--dist-labels",      default="0cm,1cm,3cm,5cm,7cm")
    parser.add_argument("--n-episodes",       default=72,    type=int)
    parser.add_argument("--eval-output-base", default="/rlwrld1/home/yashu/output/eval")
    args = parser.parse_args()

    job_ids     = [j.strip() for j in args.job_ids.split(",")]
    dist_labels = [d.strip() for d in args.dist_labels.split(",")]

    eval_job_info = {}
    for pair in args.job_map.split(","):
        pct_str, job_id = pair.split(":")
        eval_job_info[int(pct_str)] = {"job_id": job_id}

    watch(
        dataset_name     = args.dataset_name,
        job_ids          = job_ids,
        total_tasks      = args.total_tasks,
        eval_job_info    = eval_job_info,
        dist_labels      = dist_labels,
        n_episodes       = args.n_episodes,
        eval_output_base = args.eval_output_base,
        poll_interval    = args.poll_interval
    )