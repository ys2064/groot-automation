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


def notify_pipeline_complete(dataset_name: str, job_info: dict):
    lines = "\n".join([f">  `{pct}%` -> Job ID: `{info['job_id']}`" for pct, info in job_info.items()])
    _send(
        f"🎉 *Pipeline Complete!*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f"{lines}\n"
        f">  *Time:* `{_time()}`"
    )


def notify_error(phase: str, dataset_name: str, error: str):
    _send(
        f"❌ *ERROR — {phase}*\n"
        f">  *Dataset:* `{dataset_name}`\n"
        f">  *Error:*   `{error}`\n"
        f">  *Time:*    `{_time()}`"
    )
