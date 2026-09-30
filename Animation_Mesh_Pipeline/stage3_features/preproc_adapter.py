"""
The only module that touches mesh_retopo_data_preproc (upstream).

Stage 3 goes through this boundary so an upstream refactor is absorbed here.
Upstream symbols used that are not yet a public entry point (candidates to
upstream as ``preproc_pair`` / ``save_ply``):
  * data_output._save_ply
  * the label-transfer sequence of preproc.proc_data_group, which only accepts
    file paths and therefore cannot receive injected skinning data.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

# Upstream feature names whose computation reads skin weights / bone endpoints.
SKINNING_FEATURE_PREFIXES = ("skin_", "joint_", "bone_")


class Preproc:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "src" / "preproc.py").is_file():
            raise FileNotFoundError(
                f"Not a mesh_retopo_data_preproc checkout: {self.root}"
            )
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        from src import data_input, data_output, features, mathutils, metadata, preproc

        self._data_input = data_input
        self._data_output = data_output
        self._features = features
        self._mathutils = mathutils
        self._metadata = metadata
        self.label_names: list[str] = list(preproc.LABEL_NAMES)

    # --- features / metadata -------------------------------------------------
    def parse_features(self, selection: str | None) -> list[str]:
        return self._features.parse_features(selection)

    @staticmethod
    def uses_skinning(selected: list[str]) -> bool:
        return any(name.startswith(SKINNING_FEATURE_PREFIXES) for name in selected)

    def prepare_metadata(
        self, output_dir: Path, selected: list[str], with_label: bool, extra: dict
    ) -> dict:
        registry = self._features.FEATURE_REGISTRY
        data = {
            "schema_version": self._metadata.SCHEMA_VERSION,
            "format": "ascii_ply",
            "sample_domain": "face",
            "features": [
                {"name": name, "columns": list(registry[name].columns)}
                for name in selected
            ],
            "label_names": list(self.label_names) if with_label else [],
            **extra,
        }
        return self._metadata.prepare_metadata(str(output_dir), data)

    def build_features(self, mesh, selected: list[str]) -> np.ndarray:
        return self._features.build_features(mesh, selected)

    # --- meshes ----------------------------------------------------------------
    def load_mesh(self, path: Path):
        return self._data_input.load_fbx(str(path))

    @staticmethod
    def with_skinning(mesh, weights, bone_heads, bone_tails, character_scale):
        return dataclasses.replace(
            mesh,
            skin_weights=weights,
            bone_heads=bone_heads,
            bone_tails=bone_tails,
            character_scale=character_scale,
        )

    # --- labels ----------------------------------------------------------------
    def label_context(self, clean_mesh):
        from scipy.spatial import cKDTree

        mu = self._mathutils
        return (
            mu.calc_edge_flow(clean_mesh),
            mu.calc_singularity_probability(clean_mesh),
            cKDTree(mu.face_centers(clean_mesh)),
        )

    def labels(self, context, dirty_mesh) -> np.ndarray:
        mu = self._mathutils
        edge_flow, singularity, kd_tree = context
        nearest = mu.query_nearest_good_faces(dirty_mesh, kd_tree)
        labels = np.empty((dirty_mesh.num_faces, len(self.label_names)), dtype=np.float32)
        labels[:, 0:4] = mu.transfer_edge_flow(dirty_mesh, edge_flow, nearest)
        labels[:, 4] = mu.transfer_singularity_probability(singularity, nearest)
        return labels

    # --- output ------------------------------------------------------------------
    def save_ply(self, path: Path, mesh, features, labels, dataset_metadata: dict) -> None:
        expected_features = len(self._metadata.feature_names(dataset_metadata))
        expected_labels = len(dataset_metadata["label_names"])
        if features.shape != (mesh.num_faces, expected_features):
            raise ValueError(f"Feature shape {features.shape} does not match metadata")
        if expected_labels and (labels is None or labels.shape != (mesh.num_faces, expected_labels)):
            raise ValueError("Label shape does not match metadata")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".ply.tmp")
        self._data_output._save_ply(mesh, features, labels, str(tmp))
        tmp.replace(path)
