"""Per-stage run records (stageN_manifest.json) written to each stage's output root."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def git_commit(path: Path) -> str | None:
    """HEAD commit of the repo containing path, suffixed with -dirty if modified."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if status else commit


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_stage_manifest(
    output_dir: Path,
    stage: str,
    params: dict,
    counts: dict,
    started_at: str,
    extra: dict | None = None,
) -> Path:
    data = {
        "stage": stage,
        "status": "ok" if not counts.get("failed") else "partial",
        "started_at": started_at,
        "finished_at": utc_now(),
        "pipeline_commit": git_commit(PIPELINE_ROOT),
        "params": params,
        "counts": counts,
    }
    if extra:
        data.update(extra)
    path = Path(output_dir) / f"{stage}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path
