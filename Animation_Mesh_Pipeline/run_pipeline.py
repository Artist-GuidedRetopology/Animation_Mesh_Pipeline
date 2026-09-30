"""
Run pipeline stages in order from one JSON config (plain Python, no bpy).

  python run_pipeline.py --config configs/mixamo.example.json
  python run_pipeline.py --config configs/mixamo.example.json --stages 3 --force
  python run_pipeline.py --config configs/primitive.example.json --dry_run

Each stage is launched as its own Blender process with the stage's normal CLI,
so every stage can still be run by hand. Relative paths in the config are
resolved against the config file's folder.

Layout under work_dir (frames_dir / gnn_dir can be overridden in the config,
e.g. to run Stage 2+3 directly on an existing clean.fbx dataset):
  frames/   Stage 1 output, Stage 2 writes dirty*.fbx in place
  gnn/      Stage 3 output (PLY + metadata.json)
  run_state.json

A stage is skipped when its command and upstream stage are unchanged and its
last run finished with status "ok". Stage 2 (--skip_existing) and Stage 3
resume per file; Stage 1 always re-exports, so change work_dir when Stage 1
params change instead of mixing old and new samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from common.manifest import utc_now  # noqa: E402

STAGE_SCRIPTS = {
    "1": PIPELINE_ROOT / "stage1_sample" / "batch_animation_frame_sampler.py",
    "2": PIPELINE_ROOT / "stage2_dirty" / "batch_apply_dirty.py",
    "3": PIPELINE_ROOT / "stage3_features" / "build_gnn_dataset.py",
}
PATH_ARGS = {"splits", "preproc_root"}
DEFAULT_BLENDER_ARGS = {"3": ["--python-use-system-env"]}


def resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def to_flags(args: dict, base: Path) -> list[str]:
    flags: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flags.append(f"--{key}")
        if value is True:
            continue
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        elif key in PATH_ARGS:
            value = resolve(base, value)
        flags.append(str(value))
    return flags


def stage_dirs(config: dict, base: Path) -> tuple[Path, Path, Path]:
    """(work_dir, frames_dir, gnn_dir); frames_dir/gnn_dir may be overridden."""
    work_dir = resolve(base, config["work_dir"])
    frames = resolve(base, config.get("frames_dir")) or work_dir / "frames"
    gnn = resolve(base, config.get("gnn_dir")) or work_dir / "gnn"
    return work_dir, frames, gnn


def build_commands(config: dict, base: Path) -> dict[str, list[str]]:
    _, frames, gnn = stage_dirs(config, base)
    commands = {}
    for stage in ("1", "2", "3"):
        section = config.get(f"stage{stage}")
        if section is None:
            continue
        if stage == "1":
            if "input_dir" not in section:
                raise ValueError("stage1.input_dir is required")
            flags = ["--input_dir", str(resolve(base, section["input_dir"])),
                     "--output_dir", str(frames)]
        elif stage == "2":
            flags = ["--input_dir", str(frames)]
        else:
            flags = ["--input_dir", str(frames), "--output_dir", str(gnn)]
        blender_args = section.get("blender_args", DEFAULT_BLENDER_ARGS.get(stage, []))
        commands[stage] = [
            config.get("blender", "blender"),
            "--background", "--factory-startup", "--python-exit-code", "1",
            *blender_args,
            "--python", str(STAGE_SCRIPTS[stage]), "--",
            *flags,
            *to_flags(section.get("args", {}), base),
        ]
    return commands


def manifest_path(config: dict, base: Path, stage: str) -> Path:
    _, frames, gnn = stage_dirs(config, base)
    return (gnn if stage == "3" else frames) / f"stage{stage}_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default=None, help="Comma-separated subset, e.g. 2,3")
    parser.add_argument("--force", action="store_true", help="Rerun selected stages even if up to date")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commands = build_commands(config, base)
    stages = args.stages.split(",") if args.stages else config.get("stages", list(commands))
    stages = [str(s).strip() for s in stages]
    unknown = [s for s in stages if s not in commands]
    if unknown:
        raise SystemExit(f"Stages {unknown} have no section in {config_path}")

    work_dir, _, _ = stage_dirs(config, base)
    state_path = work_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    upstream_key = ""
    exit_code = 0
    for stage in ("1", "2", "3"):
        if stage not in commands:
            continue
        cmd = commands[stage]
        # Chaining the upstream finish time makes a rerun of stage N invalidate N+1.
        key = hashlib.sha256(json.dumps([cmd[1:], upstream_key]).encode()).hexdigest()
        record = state.get(stage, {})
        up_to_date = record.get("key") == key and record.get("status") == "ok"

        if stage not in stages or (up_to_date and not args.force):
            if stage in stages:
                print(f"[run_pipeline] stage {stage}: up to date, skipped")
            upstream_key = record.get("finished_at", "")
            continue

        print(f"[run_pipeline] stage {stage}: " + " ".join(cmd))
        if args.dry_run:
            upstream_key = "dry_run"
            continue

        work_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd)
        status = "failed"
        if result.returncode == 0:
            manifest = manifest_path(config, base, stage)
            status = (
                json.loads(manifest.read_text(encoding="utf-8")).get("status", "failed")
                if manifest.is_file() else "failed"
            )
        state[stage] = {"key": key, "status": status, "finished_at": utc_now()}
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"[run_pipeline] stage {stage}: {status}")
        if status != "ok":
            exit_code = 1
            if status == "failed":
                print("[run_pipeline] stopping: fix the error above and rerun")
                return exit_code
        upstream_key = state[stage]["finished_at"]
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
