"""
notify.py - Slack Notification Helper
Sends Slack messages at each pipeline phase.
"""

import os
import requests
from datetime import datetime

# ── Slack Webhook URL loaded from environment ──────────────────────────
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
# ───────────────────────────────────────────────────────────────────────


def _send(message: str):
    """Send a message to Slack."""
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as e:
        print(f"[notify] Slack notification failed: {e}")


def _time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def notify_pipeline_start(dataset_name: str, dataset_path: str, partition: str):
    _send(
        f"🚀 *GR00T Pipeline Started*\n"
        f">  *Dataset:*   `{dataset_name}`\n"
        f">  *Path:*      `{dataset_path}`\n"
        f">  *Partition:* `{partition}`\n"
        f">  *Time:*      `{_time()}`"
    )


def notify_phase1_done(dataset_name: str, split_paths: dict):
    lines = "\n".join([f">  `{pct}%` -> `{path}`" for pct, path in split_paths.items()])
    _send(
        f"✅ *Phase 1 Complete — Dataset Split*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f"{lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_phase2_done(dataset_name: str, yaml_paths: dict):
    lines = "\n".join([f">  `{pct}%` -> `{path}`" for pct, path in yaml_paths.items()])
    _send(
        f"✅ *Phase 2 Complete — YAML Configs Created*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f"{lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_phase3_done(dataset_name: str, job_info: dict):
    lines = "\n".join([f">  `{pct}%` -> Job ID: `{info['job_id']}`" for pct, info in job_info.items()])
    _send(
        f"✅ *Phase 3 Complete — SLURM Jobs Submitted*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f"{lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_training_in_progress(dataset_name: str, job_info: dict):
    lines = "\n".join([f">  `{pct}%` -> Job ID: `{info['job_id']}`" for pct, info in job_info.items()])
    _send(
        f"⏳ *Training in Progress*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f"{lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_checkpoint_saved(dataset_name: str, pct: int, step: int, checkpoint_path: str):
    _send(
        f"💾 *Checkpoint Saved — {step:,} steps*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f">  *Split:*   `{pct}%`\n"
        f">  *Path:*    `{checkpoint_path}`\n"
        f">  *Time:*    `{_time()}`"
    )


def notify_training_complete(dataset_name: str, pct: int, output_dir: str):
    import os
    try:
        checkpoints = sorted([
            f"{output_dir}/{d}" for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ])
        ckpt_lines = "\n".join([f">  `{p}`" for p in checkpoints])
    except Exception:
        ckpt_lines = f">  `{output_dir}`"

    _send(
        f"🎉 *Training Complete!*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f">  *Split:*   `{pct}%`\n"
        f">  *Checkpoints:*\n"
        f"{ckpt_lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_error(phase: str, dataset_name: str, error: str):
    _send(
        f"❌ *ERROR — {phase}*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f">  *Error:*   `{error}`\n"
        f">  *Time:*    `{_time()}`"
    )


# ── Phase 4 Eval Notifications ────────────────────────────────────────

def notify_eval_started(dataset_name: str, pct: int, dist: str):
    """
    Called from INSIDE the sbatch script after env setup.
    This means the node is actually Running (R), not Pending (PD).
    """
    _send(
        f"🤖 *Phase 4: Evaluation Started*\n"
        f">  *Dataset:*  `{dataset_name}`\n"
        f">  *Model:*    `groot{pct}`\n"
        f">  *Distance:* `{dist}`\n"
        f">  *Time:*     `{_time()}`"
    )


def notify_eval_complete(dataset_name: str, pct: int, dist: str, output_dir: str, mp4_count: int):
    """
    Called only after verifying all MP4 videos are saved.
    mp4_count is confirmed == N_EPISODES (72) before this fires.
    """
    _send(
        f"✅ *Phase 4: Evaluation Complete*\n"
        f">  *Dataset:*  `{dataset_name}`\n"
        f">  *Model:*    `groot{pct}`\n"
        f">  *Distance:* `{dist}`\n"
        f">  *Videos:*   `{mp4_count} / {mp4_count} MP4s verified ✓`\n"
        f">  *Output:*   `{output_dir}`\n"
        f">  *Time:*     `{_time()}`"
    )