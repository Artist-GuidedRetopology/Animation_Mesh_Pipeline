"""Transfer per-vertex skin weights from the clean mesh onto a dirty mesh (bpy)."""

from __future__ import annotations

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _barycentric(points, a, b, c) -> np.ndarray:
    v0, v1, v2 = b - a, c - a, points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    safe = np.abs(denom) > 1e-20
    v = np.where(safe, (d11 * d20 - d01 * d21) / np.where(safe, denom, 1), 0.0)
    w = np.where(safe, (d00 * d21 - d01 * d20) / np.where(safe, denom, 1), 0.0)
    bary = np.clip(np.column_stack((1 - v - w, v, w)), 0.0, None)
    totals = bary.sum(axis=1, keepdims=True)
    bary = np.divide(bary, totals, out=np.zeros_like(bary), where=totals > 0)
    bary[totals[:, 0] <= 0] = (1.0, 0.0, 0.0)
    return bary


def transfer_vertex_weights(
    src_vertices: np.ndarray,
    src_triangles: np.ndarray,
    src_weights: np.ndarray,
    dst_vertices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Project each dst vertex onto the closest src triangle and interpolate the
    triangle's vertex weights barycentrically.

    Returns (dst_weights, max projection distance).
    """
    tree = BVHTree.FromPolygons(
        [tuple(v) for v in src_vertices],
        [tuple(int(i) for i in t) for t in src_triangles],
        all_triangles=True,
    )
    locations = np.empty((len(dst_vertices), 3), dtype=np.float64)
    triangle_ids = np.empty(len(dst_vertices), dtype=np.int64)
    distances = np.empty(len(dst_vertices), dtype=np.float64)
    for i, co in enumerate(dst_vertices):
        location, _normal, index, distance = tree.find_nearest(Vector(co))
        if index is None:
            raise RuntimeError(f"No clean triangle found for dirty vertex {i}")
        locations[i] = location
        triangle_ids[i] = index
        distances[i] = distance

    corners = src_triangles[triangle_ids]
    bary = _barycentric(
        locations,
        src_vertices[corners[:, 0]],
        src_vertices[corners[:, 1]],
        src_vertices[corners[:, 2]],
    )
    weights = np.einsum("ik,ikb->ib", bary, src_weights[corners])
    return weights, float(distances.max(initial=0.0))
