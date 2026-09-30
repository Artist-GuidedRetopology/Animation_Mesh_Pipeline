"""
Skinning sidecar files written by Stage 1 and consumed by Stage 3.

skin.npz  (one per character, next to the sample folders)
    Bind-pose skin weights for the vertices of the exported clean mesh, stored
    sparsely as the top-K influences per vertex, plus rest-pose bone endpoints
    and a pose-invariant character scale.

pose.npz  (one per sample, next to clean.fbx)
    World-space bone endpoints at the sampled frame, plus a few vertex
    positions used to verify that clean.fbx vertex order and coordinate frame
    still match skin.npz after the FBX round trip.

Bone columns are the character armature's deform bones in armature order and
are identical in both files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
SKIN_FILENAME = "skin.npz"
POSE_FILENAME = "pose.npz"
MAX_INFLUENCES = 8
NUM_CHECK_VERTICES = 64


@dataclass
class SkinSidecar:
    bone_names: list[str]
    weight_indices: np.ndarray  # (V, K) int32
    weight_values: np.ndarray  # (V, K) float32
    rest_bone_heads: np.ndarray  # (B, 3)
    rest_bone_tails: np.ndarray  # (B, 3)
    character_scale: float
    object_names: list[str]
    object_vertex_offsets: np.ndarray  # (num_objects + 1,)

    @property
    def vertex_count(self) -> int:
        return int(self.weight_indices.shape[0])

    def dense_weights(self) -> np.ndarray:
        weights = np.zeros((self.vertex_count, len(self.bone_names)), dtype=np.float64)
        rows = np.arange(self.vertex_count)[:, None]
        np.add.at(weights, (rows, self.weight_indices), self.weight_values)
        return weights


@dataclass
class PoseSidecar:
    bone_names: list[str]
    bone_heads: np.ndarray  # (B, 3)
    bone_tails: np.ndarray  # (B, 3)
    frame: int
    animation: str
    check_indices: np.ndarray  # (N,)
    check_positions: np.ndarray  # (N, 3)


def check_vertex_indices(vertex_count: int) -> np.ndarray:
    count = min(NUM_CHECK_VERTICES, vertex_count)
    return np.unique(np.linspace(0, vertex_count - 1, count).astype(np.int64))


def sparse_top_k(
    per_vertex: list[list[tuple[int, float]]], k: int = MAX_INFLUENCES
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros((len(per_vertex), k), dtype=np.int32)
    values = np.zeros((len(per_vertex), k), dtype=np.float32)
    for row, influences in enumerate(per_vertex):
        top = sorted(influences, key=lambda item: item[1], reverse=True)[:k]
        for col, (bone, weight) in enumerate(top):
            indices[row, col] = bone
            values[row, col] = weight
    return indices, values


def save_skin(path: Path, skin: SkinSidecar) -> None:
    np.savez_compressed(
        path,
        format_version=FORMAT_VERSION,
        bone_names=np.asarray(skin.bone_names, dtype=str),
        weight_indices=skin.weight_indices.astype(np.int32),
        weight_values=skin.weight_values.astype(np.float32),
        rest_bone_heads=np.asarray(skin.rest_bone_heads, dtype=np.float64),
        rest_bone_tails=np.asarray(skin.rest_bone_tails, dtype=np.float64),
        character_scale=float(skin.character_scale),
        object_names=np.asarray(skin.object_names, dtype=str),
        object_vertex_offsets=np.asarray(skin.object_vertex_offsets, dtype=np.int64),
    )


def load_skin(path: Path) -> SkinSidecar:
    with np.load(path, allow_pickle=False) as data:
        _check_version(path, data)
        return SkinSidecar(
            bone_names=[str(n) for n in data["bone_names"]],
            weight_indices=data["weight_indices"],
            weight_values=data["weight_values"],
            rest_bone_heads=data["rest_bone_heads"],
            rest_bone_tails=data["rest_bone_tails"],
            character_scale=float(data["character_scale"]),
            object_names=[str(n) for n in data["object_names"]],
            object_vertex_offsets=data["object_vertex_offsets"],
        )


def save_pose(path: Path, pose: PoseSidecar) -> None:
    np.savez_compressed(
        path,
        format_version=FORMAT_VERSION,
        bone_names=np.asarray(pose.bone_names, dtype=str),
        bone_heads=np.asarray(pose.bone_heads, dtype=np.float64),
        bone_tails=np.asarray(pose.bone_tails, dtype=np.float64),
        frame=int(pose.frame),
        animation=str(pose.animation),
        check_indices=np.asarray(pose.check_indices, dtype=np.int64),
        check_positions=np.asarray(pose.check_positions, dtype=np.float64),
    )


def load_pose(path: Path) -> PoseSidecar:
    with np.load(path, allow_pickle=False) as data:
        _check_version(path, data)
        return PoseSidecar(
            bone_names=[str(n) for n in data["bone_names"]],
            bone_heads=data["bone_heads"],
            bone_tails=data["bone_tails"],
            frame=int(data["frame"]),
            animation=str(data["animation"]),
            check_indices=data["check_indices"],
            check_positions=data["check_positions"],
        )


def find_skin_file(sample_dir: Path, root: Path) -> Path | None:
    """Nearest skin.npz from sample_dir up to (and including) root."""
    sample_dir = sample_dir.resolve()
    root = root.resolve()
    for directory in (sample_dir, *sample_dir.parents):
        candidate = directory / SKIN_FILENAME
        if candidate.is_file():
            return candidate
        if directory == root:
            break
    return None


def _check_version(path: Path, data) -> None:
    version = int(data["format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported sidecar format_version {version}")
