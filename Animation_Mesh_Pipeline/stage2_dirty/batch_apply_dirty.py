"""
Batch apply dirty topology to previously sampled clean meshes (bpy).

递归扫描 input_dir 下的 clean.*，在同目录按强度写出 dirty。

默认每帧导出多档强度：
  dirty10.fbx / dirty20.fbx / dirty40.fbx / dirty70.fbx / dirty100.fbx

用法:
  Blender --background --python stage2_dirty/batch_apply_dirty.py -- \\
    --input_dir "/path/to/output_animation_frames" \\
    --displace_strengths 10,20,40,70,100
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import random
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dirty_topology_core
from common.manifest import utc_now, write_stage_manifest

importlib.reload(dirty_topology_core)
from dirty_topology_core import DirtyParams, apply_dirty_topology_to_mesh

# =========================
# 默认可调参数
# =========================
# 每帧各强度各出 1 份；文件名 dirty{强度}.fbx
DEFAULT_DISPLACE_STRENGTHS = (10.0, 20.0, 40.0, 70.0, 100.0)
EXPORT_FORMAT = "fbx"  # 仅处理同格式文件；输出同格式
RANDOM_SEED = 42
PARAMS = DirtyParams(
    displace_scale=0.002,
    displace_strength=100.0,
    enable_random_merge=True,
    merge_edge_ratio=0.05,
    enable_local_disease=True,
    disease_centers=2,
    disease_ring=2,
    disease_subdivide=True,
    disease_collapse_ratio=0.08,
)

RUN_LOG_PATH: str | None = None


def log(msg: str) -> None:
    text = f"[BatchDirty] {msg}"
    print(text)
    if RUN_LOG_PATH:
        try:
            with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        except Exception:
            break


def import_mesh_file(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects.keys())
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
    elif suffix == ".obj":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise ValueError(f"Unsupported format: {path}")
    after = set(bpy.data.objects.keys())
    return [
        bpy.data.objects[n]
        for n in (after - before)
        if n in bpy.data.objects and bpy.data.objects[n].type == "MESH"
    ]


def export_selected_meshes(export_path: Path, export_format: str) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    filepath = str(export_path)
    if export_format == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            use_selection=True,
            object_types={"MESH"},
            add_leaf_bones=False,
            bake_anim=False,
            axis_forward="-Y",
            axis_up="Z",
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_NONE",
        )
    elif export_format == "obj":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(
                filepath=filepath,
                export_selected_objects=True,
                export_uv=False,
                export_normals=True,
                export_materials=False,
            )
        else:
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                use_materials=False,
                use_uvs=False,
                use_normals=True,
            )
    elif export_format == "ply":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "ply_export"):
            bpy.ops.wm.ply_export(
                filepath=filepath,
                export_selected_objects=True,
                export_uv=False,
                export_normals=True,
            )
        else:
            bpy.ops.export_mesh.ply(
                filepath=filepath,
                use_selection=True,
                use_uvs=False,
                use_normals=True,
            )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def find_clean_files(input_dir: Path, export_format: str) -> list[Path]:
    return sorted(input_dir.rglob(f"clean.{export_format}"))


def strength_label(strength: float) -> str:
    """10 -> '10'; 10.5 -> '10_5'."""
    if float(strength).is_integer():
        return str(int(strength))
    return str(strength).replace(".", "_")


def parse_strengths(raw: str | None) -> list[float]:
    if raw is None or not str(raw).strip():
        return list(DEFAULT_DISPLACE_STRENGTHS)
    strengths: list[float] = []
    for part in str(raw).replace(" ", "").split(","):
        if not part:
            continue
        strengths.append(float(part))
    if not strengths:
        raise ValueError("displace_strengths is empty")
    return strengths


def process_clean_file(
    clean_path: Path,
    strengths: list[float],
    base_params: DirtyParams,
    export_format: str,
    skip_existing: bool = False,
    random_seed: int = RANDOM_SEED,
) -> int:
    jobs = []
    for strength in strengths:
        label = strength_label(strength)
        out_path = clean_path.parent / f"dirty{label}.{export_format}"
        if not (skip_existing and out_path.is_file()):
            jobs.append((strength, out_path))
    if not jobs:
        return 0

    clear_scene()
    sources = import_mesh_file(clean_path)
    if not sources:
        raise RuntimeError(f"No mesh imported from: {clean_path}")
    for src in sources:
        src.name = "clean_source"

    collection = bpy.context.scene.collection
    written = 0
    for strength, out_path in jobs:
        # Stable per-output seed makes interruption/resume reproducible.
        seed_key = f"{random_seed}:{clean_path.as_posix()}:{float(strength)}"
        file_seed = int.from_bytes(
            hashlib.sha256(seed_key.encode("utf-8")).digest()[:8],
            byteorder="little",
        )
        random.seed(file_seed)
        params = replace(base_params, displace_strength=float(strength))

        dirty_objects = []
        for src in sources:
            obj = src.copy()
            obj.data = src.data.copy()
            collection.objects.link(obj)
            apply_dirty_topology_to_mesh(obj.data, params)
            obj.name = "mesh"
            dirty_objects.append(obj)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in dirty_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = dirty_objects[0]

        export_selected_meshes(out_path, export_format)
        written += 1

        for obj in dirty_objects:
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)

    return written


def run_pipeline(
    input_dir: str,
    displace_strengths: list[float] | None = None,
    export_format: str = EXPORT_FORMAT,
    params: DirtyParams | None = None,
    random_seed: int = RANDOM_SEED,
    skip_existing: bool = False,
) -> None:
    started_at = utc_now()
    in_path = Path(bpy.path.abspath(input_dir)).expanduser().resolve()
    if not in_path.is_dir():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {in_path}")

    params = params or PARAMS
    random.seed(int(random_seed))
    strengths = (
        list(displace_strengths)
        if displace_strengths is not None
        else list(DEFAULT_DISPLACE_STRENGTHS)
    )
    export_format = export_format.lower().strip(".")
    global RUN_LOG_PATH
    RUN_LOG_PATH = str(in_path / "dirty_run_log.txt")
    error_log = in_path / "dirty_error_log.txt"

    clean_files = find_clean_files(in_path, export_format)
    if not clean_files:
        raise FileNotFoundError(
            f"No clean.{export_format} found under: {in_path}"
        )

    log("Batch dirty started")
    log(f"INPUT_DIR={in_path}")
    log(f"clean files={len(clean_files)}")
    log(f"displace_strengths(%)={strengths}")
    log(
        f"displace_scale={params.displace_scale}, "
        f"merge={params.enable_random_merge}@{params.merge_edge_ratio}, "
        f"disease={params.enable_local_disease} centers={params.disease_centers}, "
        f"seed={random_seed}"
    )

    ok = 0
    total_written = 0
    for idx, clean_path in enumerate(clean_files, start=1):
        try:
            log(f"[{idx}/{len(clean_files)}] {clean_path.relative_to(in_path)}")
            total_written += process_clean_file(
                clean_path,
                strengths,
                params,
                export_format,
                skip_existing=skip_existing,
                random_seed=random_seed,
            )
            ok += 1
        except Exception as exc:
            tb = traceback.format_exc()
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"{clean_path}\n{exc}\n{tb}\n{'-' * 80}\n")
            log(f"  Failed: {exc}")

    summary = (
        f"Done: {ok}/{len(clean_files)} clean files ok, "
        f"{total_written} dirty files written "
        f"({len(strengths)} strengths × {ok} frames)."
    )
    log(summary)
    write_stage_manifest(
        in_path,
        "stage2",
        params={
            "displace_strengths": strengths,
            "export_format": export_format,
            "random_seed": random_seed,
            "skip_existing": skip_existing,
            "dirty_params": asdict(params),
        },
        counts={
            "clean_files": len(clean_files),
            "clean_ok": ok,
            "dirty_written": total_written,
            "failed": len(clean_files) - ok,
        },
        started_at=started_at,
    )


def parse_args():
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    default_strengths = ",".join(str(int(s)) for s in DEFAULT_DISPLACE_STRENGTHS)
    parser = argparse.ArgumentParser(description="Batch dirty topology on clean meshes")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--displace_strengths",
        default=default_strengths,
        help="Comma-separated displace strengths in percent, e.g. 10,20,40,70,100",
    )
    parser.add_argument(
        "--export_format", choices=("fbx", "obj", "ply"), default=EXPORT_FORMAT
    )
    parser.add_argument("--displace_scale", type=float, default=PARAMS.displace_scale)
    parser.add_argument(
        "--merge_edge_ratio", type=float, default=PARAMS.merge_edge_ratio
    )
    parser.add_argument("--disease_centers", type=int, default=PARAMS.disease_centers)
    parser.add_argument(
        "--disease_collapse_ratio",
        type=float,
        default=PARAMS.disease_collapse_ratio,
    )
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument("--no_disease", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Keep existing dirty files and only generate missing outputs",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    strengths = parse_strengths(args.displace_strengths)
    params = DirtyParams(
        displace_scale=args.displace_scale,
        displace_strength=strengths[0],
        enable_random_merge=not args.no_merge,
        merge_edge_ratio=args.merge_edge_ratio,
        enable_local_disease=not args.no_disease,
        disease_centers=args.disease_centers,
        disease_collapse_ratio=args.disease_collapse_ratio,
        disease_ring=PARAMS.disease_ring,
        disease_subdivide=PARAMS.disease_subdivide,
        enable_edge_rotate=PARAMS.enable_edge_rotate,
        rotate_edge_ratio=PARAMS.rotate_edge_ratio,
        merge_distance=PARAMS.merge_distance,
    )
    run_pipeline(
        input_dir=args.input_dir,
        displace_strengths=strengths,
        export_format=args.export_format,
        params=params,
        random_seed=args.seed,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__" or "CTX" in globals():
    try:
        if "--" in sys.argv:
            main()
        else:
            default = (
                SCRIPT_DIR.parent.parent.parent / "Data" / "output_animation_frames"
            )
            raw = input(f"Enter input_dir [{default}]: ").strip()
            input_dir = Path(raw).expanduser().resolve() if raw else default
            run_pipeline(str(input_dir))
    except Exception as exc:
        log(f"Fatal error: {exc}")
        traceback.print_exc()
        raise
