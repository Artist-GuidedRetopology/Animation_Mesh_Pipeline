"""
Stage 3: clean/dirty FBX samples -> per-face feature/label ASCII PLY for the GNN (bpy).

Input  (Stage 2 output, any nesting depth):
  <input_dir>/<...>/<sample>/clean.fbx
  <input_dir>/<...>/<sample>/dirty*.fbx
  <input_dir>/<...>/<sample>/pose.npz       # only needed for skinning features
  <input_dir>/<char>/skin.npz               # only needed for skinning features

Output (relative layout mirrored, optionally grouped by split):
  <output_dir>/metadata.json
  <output_dir>/[<split>/]<...>/<sample>/dirty*.ply
  <output_dir>/stage3_manifest.json

Features and labels are computed by mesh_retopo_data_preproc. Skinning
features use Stage 1 sidecars: clean weights map 1:1 onto clean vertices,
dirty weights are interpolated from the closest clean triangle, and bone
endpoints come from the sample's own frame.

Usage:
  Blender --background --python-use-system-env \\
    --python stage3_features/build_gnn_dataset.py -- \\
    --input_dir ".../frames" --output_dir ".../gnn" \\
    [--features "position,normal,skin_entropy"] [--splits splits.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, PIPELINE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import skin_sidecar  # noqa: E402
from common.manifest import git_commit, utc_now, write_stage_manifest  # noqa: E402
from preproc_adapter import Preproc  # noqa: E402
from skin_transfer import transfer_vertex_weights  # noqa: E402

# Scripts/Animation_Mesh_Pipeline/Animation_Mesh_Pipeline -> Artist-Guided_Retopology
REPO_PARENT = PIPELINE_ROOT.parents[3]
DEFAULT_PREPROC_ROOT = (
    REPO_PARENT / "lab" / "perface-data-preproc-pipeline" / "mesh_retopo_data_preproc"
)
CLEAN_NAME = "clean.fbx"
DIRTY_GLOB = "dirty*.fbx"
# pose.npz check vertices must survive the FBX round trip within this fraction
# of the character scale; larger errors mean vertex order or frame changed.
ROUND_TRIP_TOLERANCE = 1e-4

RUN_LOG_PATH: Path | None = None


def log(msg: str) -> None:
    text = f"[BuildGNN] {msg}"
    print(text)
    if RUN_LOG_PATH is not None:
        with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def load_splits(path: Path | None) -> dict[str, str] | None:
    """splits.json {"train": [...], "val": [...], "test": [...]} -> {key: split}."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for split, keys in data.items():
        if not isinstance(keys, list):
            continue
        for key in keys:
            if key in lookup:
                raise ValueError(f"{path}: {key!r} is in both {lookup[key]} and {split}")
            lookup[key] = split
    return lookup


class SkinningSource:
    """Loads and validates Stage 1 sidecars for one sample."""

    def __init__(self, input_root: Path):
        self.input_root = input_root
        self._skins: dict[Path, skin_sidecar.SkinSidecar] = {}
        self._dense: dict[Path, np.ndarray] = {}

    def for_sample(self, sample_dir: Path, clean_mesh):
        skin_path = skin_sidecar.find_skin_file(sample_dir, self.input_root)
        if skin_path is None:
            raise FileNotFoundError(f"No {skin_sidecar.SKIN_FILENAME} above {sample_dir}")
        if skin_path not in self._skins:
            self._skins[skin_path] = skin_sidecar.load_skin(skin_path)
            self._dense[skin_path] = self._skins[skin_path].dense_weights()
        skin = self._skins[skin_path]
        pose = skin_sidecar.load_pose(sample_dir / skin_sidecar.POSE_FILENAME)

        if len(clean_mesh.vertices) != skin.vertex_count:
            raise ValueError(
                f"clean.fbx has {len(clean_mesh.vertices)} verts, "
                f"{skin_path.name} has {skin.vertex_count}"
            )
        if pose.bone_names != skin.bone_names:
            raise ValueError("pose.npz and skin.npz bone columns differ")
        error = np.abs(clean_mesh.vertices[pose.check_indices] - pose.check_positions).max()
        if error > ROUND_TRIP_TOLERANCE * skin.character_scale:
            raise ValueError(
                f"clean.fbx vertex order/frame does not match pose.npz (max error {error:.3g})"
            )
        return self._dense[skin_path], pose, skin.character_scale


def run_pipeline(
    input_dir: str,
    output_dir: str,
    preproc_root: str = str(DEFAULT_PREPROC_ROOT),
    features: str | None = None,
    with_label: bool = True,
    splits_path: str | None = None,
    overwrite: bool = False,
    max_samples: int | None = None,
) -> None:
    global RUN_LOG_PATH
    started_at = utc_now()
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    if not in_root.is_dir():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH = out_root / "stage3_run_log.txt"
    error_log = out_root / "stage3_error_log.txt"

    pp = Preproc(Path(preproc_root))
    selected = pp.parse_features(features)
    uses_skin = pp.uses_skinning(selected)
    splits = load_splits(Path(splits_path).expanduser().resolve() if splits_path else None)
    dataset_metadata = pp.prepare_metadata(
        out_root,
        selected,
        with_label,
        extra={
            "pipeline": {
                "skin_source": "stage1_sidecar_barycentric" if uses_skin else None,
                "splits": splits_path,
            }
        },
    )

    clean_files = sorted(in_root.rglob(CLEAN_NAME))
    if max_samples is not None:
        clean_files = clean_files[:max_samples]
    if not clean_files:
        raise FileNotFoundError(f"No {CLEAN_NAME} found under: {in_root}")

    log("Stage 3 started")
    log(f"INPUT_DIR={in_root}")
    log(f"OUTPUT_DIR={out_root}")
    log(f"PREPROC_ROOT={pp.root} ({git_commit(pp.root)})")
    log(f"features={selected}, labels={with_label}, skinning={uses_skin}")
    log(f"samples={len(clean_files)}, splits={'on' if splits else 'off'}")

    skinning = SkinningSource(in_root) if uses_skin else None
    counts = {
        "samples": len(clean_files), "written": 0, "skipped_existing": 0,
        "no_dirty": 0, "unassigned_split": 0, "failed": 0,
    }
    split_counts: dict[str, int] = {}
    max_transfer_distance = 0.0

    for idx, clean_path in enumerate(clean_files, start=1):
        sample_dir = clean_path.parent
        rel = sample_dir.relative_to(in_root)
        split = None
        if splits is not None:
            split = splits.get(rel.parts[0]) if rel.parts else None
            if split is None:
                counts["unassigned_split"] += 1
                continue
        target_dir = out_root / split / rel if split else out_root / rel

        dirty_paths = sorted(sample_dir.glob(DIRTY_GLOB))
        if not dirty_paths:
            counts["no_dirty"] += 1
            continue
        pending = [
            (p, target_dir / f"{p.stem}.ply") for p in dirty_paths
            if overwrite or not (target_dir / f"{p.stem}.ply").is_file()
        ]
        counts["skipped_existing"] += len(dirty_paths) - len(pending)
        if not pending:
            continue

        try:
            log(f"[{idx}/{len(clean_files)}] {rel} ({len(pending)} dirty)")
            clean = pp.load_mesh(clean_path)
            clean_weights = pose = scale = None
            if skinning is not None:
                clean_weights, pose, scale = skinning.for_sample(sample_dir, clean)
            label_context = pp.label_context(clean) if with_label else None

            for dirty_path, ply_path in pending:
                dirty = pp.load_mesh(dirty_path)
                if skinning is not None:
                    weights, distance = transfer_vertex_weights(
                        clean.vertices, clean.triangles, clean_weights, dirty.vertices
                    )
                    max_transfer_distance = max(max_transfer_distance, distance / scale)
                    dirty = pp.with_skinning(
                        dirty, weights, pose.bone_heads, pose.bone_tails, scale
                    )
                face_features = pp.build_features(dirty, selected)
                labels = pp.labels(label_context, dirty) if with_label else None
                pp.save_ply(ply_path, dirty, face_features, labels, dataset_metadata)
                counts["written"] += 1
                if split:
                    split_counts[split] = split_counts.get(split, 0) + 1
        except Exception as exc:
            counts["failed"] += 1
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"{sample_dir}\n{exc}\n{traceback.format_exc()}\n{'-' * 80}\n")
            log(f"  Failed: {exc}")

    log(
        f"Done: {counts['written']} PLY written, {counts['skipped_existing']} existing, "
        f"{counts['failed']} failed samples, {counts['no_dirty']} without dirty, "
        f"{counts['unassigned_split']} not in splits"
    )
    if uses_skin:
        log(f"max dirty->clean projection distance: {max_transfer_distance:.3g} x character scale")
    write_stage_manifest(
        out_root,
        "stage3",
        params={
            "input_dir": str(in_root),
            "features": selected,
            "with_label": with_label,
            "splits": splits_path,
            "overwrite": overwrite,
            "max_samples": max_samples,
        },
        counts={**counts, "per_split": split_counts},
        started_at=started_at,
        extra={
            "preproc_root": str(pp.root),
            "preproc_commit": git_commit(pp.root),
            "max_skin_transfer_distance_rel": max_transfer_distance if uses_skin else None,
        },
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: build GNN feature/label PLYs")
    parser.add_argument("--input_dir", required=True, help="Stage 2 output root")
    parser.add_argument("--output_dir", required=True, help="PLY dataset root")
    parser.add_argument("--preproc_root", default=str(DEFAULT_PREPROC_ROOT))
    parser.add_argument(
        "--features", default=None,
        help="Comma-separated upstream feature names (default: upstream defaults)",
    )
    parser.add_argument("--no_label", action="store_true")
    parser.add_argument("--splits", default=None, help="splits.json keyed by top-level folder")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing PLYs")
    parser.add_argument("--max_samples", type=int, default=None, help="Smoke-test limit")
    return parser.parse_args(argv)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preproc_root=args.preproc_root,
        features=args.features,
        with_label=not args.no_label,
        splits_path=args.splits,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
