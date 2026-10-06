"""
Dirty topology core (bmesh pipeline).

Pipeline:
  1. Triangulate
  2. Tangent-plane (local XY) random displace
  3. Random edge merge / collapse
  4. Local disease (optional)

All knobs are module-level; override before calling, or pass DirtyParams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import bmesh
from mathutils import Vector

if TYPE_CHECKING:
    import bpy.types


@dataclass
class DirtyParams:
    # --- displace (tangent plane) ---
    # magnitude = bbox_diagonal * displace_scale * (displace_strength / 100)
    # displace_strength=100 → 100% of base scale; 50 → half
    displace_scale: float = 0.002
    displace_strength: float = 100.0  # percent, 0–100+

    # --- random merge ---
    enable_random_merge: bool = True
    merge_edge_ratio: float = 0.05  # fraction of edges to collapse
    merge_distance: float = 1e-6  # remove_doubles cleanup

    # --- local disease ---
    enable_local_disease: bool = True
    disease_centers: int = 2
    disease_ring: int = 2  # n-ring neighborhood
    disease_subdivide: bool = True
    disease_collapse_ratio: float = 0.08

    # --- optional extras ---
    enable_edge_rotate: bool = False
    rotate_edge_ratio: float = 0.02


# Default singleton used when callers don't pass params
DEFAULT_PARAMS = DirtyParams()


def mesh_bbox_diagonal(bm: bmesh.types.BMesh) -> float:
    if not bm.verts:
        return 1.0
    coords = [vert.co for vert in bm.verts]
    min_co = Vector(
        (min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords))
    )
    max_co = Vector(
        (max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords))
    )
    diagonal = (max_co - min_co).length
    return diagonal if diagonal > 1e-8 else 1.0


def refresh_bmesh_indices(bm: bmesh.types.BMesh) -> None:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


def valid_verts(bm: bmesh.types.BMesh) -> list[bmesh.types.BMVert]:
    refresh_bmesh_indices(bm)
    return [vert for vert in bm.verts if vert.is_valid]


def valid_edges(bm: bmesh.types.BMesh) -> list[bmesh.types.BMEdge]:
    refresh_bmesh_indices(bm)
    return [edge for edge in bm.edges if edge.is_valid]


def valid_faces(bm: bmesh.types.BMesh) -> list[bmesh.types.BMFace]:
    refresh_bmesh_indices(bm)
    return [face for face in bm.faces if face.is_valid]


def random_edge_subset(
    edges: list[bmesh.types.BMEdge], ratio: float
) -> list[bmesh.types.BMEdge]:
    if not edges or ratio <= 0:
        return []
    count = max(1, int(len(edges) * ratio))
    count = min(count, len(edges))
    return random.sample(edges, count)


def triangulate_all(bm: bmesh.types.BMesh) -> None:
    faces = valid_faces(bm)
    if faces:
        bmesh.ops.triangulate(bm, faces=faces)
        refresh_bmesh_indices(bm)


def _tangent_basis(normal: Vector) -> tuple[Vector, Vector]:
    """Build orthonormal (tangent, bitangent) spanning the plane perpendicular to normal."""
    n = normal.normalized()
    if n.length < 1e-12:
        n = Vector((0.0, 0.0, 1.0))

    # Pick a helper axis least aligned with n
    helper = Vector((1.0, 0.0, 0.0))
    if abs(n.dot(helper)) > 0.9:
        helper = Vector((0.0, 1.0, 0.0))

    tangent = n.cross(helper).normalized()
    bitangent = n.cross(tangent).normalized()
    return tangent, bitangent


def displace_tangent_plane(bm: bmesh.types.BMesh, magnitude: float) -> None:
    """Random displace each vertex in its local tangent plane (surface XY)."""
    if magnitude <= 0:
        return

    for vert in valid_verts(bm):
        normal = vert.normal
        if normal.length < 1e-12:
            # Fallback: average linked face normals
            n = Vector((0.0, 0.0, 0.0))
            for face in vert.link_faces:
                if face.is_valid:
                    n += face.normal
            if n.length < 1e-12:
                continue
            normal = n.normalized()

        tangent, bitangent = _tangent_basis(normal)
        u = random.uniform(-magnitude, magnitude)
        v = random.uniform(-magnitude, magnitude)
        vert.co += tangent * u + bitangent * v


def _independent_edge_batch(
    edges: list[bmesh.types.BMEdge],
) -> tuple[list[bmesh.types.BMEdge], list[bmesh.types.BMEdge]]:
    """Split edges into a vertex-disjoint batch and the rest."""
    used: set[bmesh.types.BMVert] = set()
    batch: list[bmesh.types.BMEdge] = []
    rest: list[bmesh.types.BMEdge] = []
    for edge in edges:
        if not edge.is_valid:
            continue
        a, b = edge.verts
        if a in used or b in used:
            rest.append(edge)
            continue
        used.update((a, b))
        batch.append(edge)
    return batch, rest


def collapse_edges_robust(
    bm: bmesh.types.BMesh, edges: list[bmesh.types.BMEdge], max_rounds: int = 8
) -> int:
    # Each bmesh.ops call costs O(mesh size), so collapse vertex-disjoint
    # batches (same result as one-by-one) instead of one edge per call.
    collapsed_count = 0
    pending = list(edges)
    for _ in range(max_rounds):
        batch, pending = _independent_edge_batch(pending)
        if not batch:
            break
        try:
            bmesh.ops.collapse(bm, edges=batch, uvs=False)
            collapsed_count += len(batch)
        except Exception:
            for edge in batch:
                if not edge.is_valid:
                    continue
                try:
                    bmesh.ops.collapse(bm, edges=[edge], uvs=False)
                    collapsed_count += 1
                except Exception:
                    continue
        pending = [edge for edge in pending if edge.is_valid]
    refresh_bmesh_indices(bm)
    return collapsed_count


def rotate_edges_robust(
    bm: bmesh.types.BMesh, edges: list[bmesh.types.BMEdge]
) -> int:
    rotated_count = 0
    for edge in edges:
        if not edge.is_valid or len(edge.link_faces) != 2:
            continue
        try:
            new_edge = bmesh.utils.edge_rotate(
                edge, ccw=random.choice([True, False])
            )
        except Exception:
            new_edge = None
        if new_edge is not None:
            rotated_count += 1
    refresh_bmesh_indices(bm)
    return rotated_count


def cleanup_mesh_light(bm: bmesh.types.BMesh, merge_distance: float) -> int:
    merged = 0
    verts = valid_verts(bm)
    if verts and merge_distance > 0:
        result = bmesh.ops.remove_doubles(bm, verts=verts, dist=merge_distance)
        if isinstance(result, dict):
            merged = len(result.get("verts", []))
        refresh_bmesh_indices(bm)

    loose_verts = [v for v in valid_verts(bm) if not v.link_edges]
    loose_edges = [e for e in valid_edges(bm) if not e.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
        refresh_bmesh_indices(bm)
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        refresh_bmesh_indices(bm)
    return merged


def find_nearest_vert(
    bm: bmesh.types.BMesh, target_co: Vector
) -> bmesh.types.BMVert | None:
    bm.verts.ensure_lookup_table()
    best_vert = None
    best_dist_sq = float("inf")
    for vert in bm.verts:
        if not vert.is_valid:
            continue
        dist_sq = (vert.co - target_co).length_squared
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_vert = vert
    return best_vert


def collect_n_ring_faces(
    bm: bmesh.types.BMesh, center_vert: bmesh.types.BMVert, rings: int
) -> list[bmesh.types.BMFace]:
    if not center_vert.is_valid or rings <= 0:
        return []

    refresh_bmesh_indices(bm)
    visited_verts = {center_vert}
    frontier = {center_vert}

    for _ in range(rings):
        next_frontier: set[bmesh.types.BMVert] = set()
        for vert in frontier:
            if not vert.is_valid:
                continue
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other.is_valid and other not in visited_verts:
                    visited_verts.add(other)
                    next_frontier.add(other)
        frontier = next_frontier

    faces: set[bmesh.types.BMFace] = set()
    for vert in visited_verts:
        if vert.is_valid:
            faces.update(face for face in vert.link_faces if face.is_valid)
    return list(faces)


def faces_to_edges(faces: list[bmesh.types.BMFace]) -> list[bmesh.types.BMEdge]:
    edges: set[bmesh.types.BMEdge] = set()
    for face in faces:
        if face.is_valid:
            edges.update(edge for edge in face.edges if edge.is_valid)
    return list(edges)


def is_all_triangle_mesh(bm: bmesh.types.BMesh) -> bool:
    faces = valid_faces(bm)
    return bool(faces) and all(len(face.verts) == 3 for face in faces)


def apply_dirty_topology_to_bmesh(
    bm: bmesh.types.BMesh,
    params: DirtyParams | None = None,
) -> dict[str, int | bool | float]:
    """Apply dirty topology in-place; return stats."""
    p = params or DEFAULT_PARAMS
    stats: dict[str, int | bool | float] = {
        "collapsed_edges": 0,
        "rotated_edges": 0,
        "disease_center_count": 0,
        "merged_verts": 0,
        "displace_magnitude": 0.0,
        "all_triangles": False,
    }

    # 1) Triangulate first
    triangulate_all(bm)

    # 2) Tangent-plane XY displace
    strength = max(0.0, float(p.displace_strength)) / 100.0
    magnitude = mesh_bbox_diagonal(bm) * max(0.0, float(p.displace_scale)) * strength
    stats["displace_magnitude"] = magnitude
    displace_tangent_plane(bm, magnitude)

    # 3) Random merge (edge collapse)
    if p.enable_random_merge and p.merge_edge_ratio > 0:
        edges_to_collapse = random_edge_subset(valid_edges(bm), p.merge_edge_ratio)
        stats["collapsed_edges"] = collapse_edges_robust(bm, edges_to_collapse)
        triangulate_all(bm)  # collapse can create non-tris

    # 4) Local disease
    if p.enable_local_disease and p.disease_centers > 0:
        verts = valid_verts(bm)
        if verts:
            center_count = min(max(int(p.disease_centers), 1), len(verts))
            stats["disease_center_count"] = center_count
            center_coords = [
                vert.co.copy() for vert in random.sample(verts, center_count)
            ]

            for center_co in center_coords:
                center = find_nearest_vert(bm, center_co)
                if center is None:
                    continue

                local_faces = collect_n_ring_faces(bm, center, max(1, int(p.disease_ring)))
                local_edges = faces_to_edges(local_faces)

                if p.disease_subdivide and local_edges:
                    bmesh.ops.subdivide_edges(
                        bm,
                        edges=local_edges,
                        cuts=1,
                        use_grid_fill=True,
                    )
                    refresh_bmesh_indices(bm)

                center = find_nearest_vert(bm, center_co)
                if center is None:
                    continue

                refreshed_faces = collect_n_ring_faces(
                    bm, center, max(1, int(p.disease_ring))
                )
                refreshed_edges = faces_to_edges(refreshed_faces)
                local_collapse = random_edge_subset(
                    refreshed_edges, p.disease_collapse_ratio
                )
                stats["collapsed_edges"] += collapse_edges_robust(bm, local_collapse)

            triangulate_all(bm)

    if p.enable_edge_rotate and p.rotate_edge_ratio > 0:
        internal_edges = [
            edge for edge in valid_edges(bm) if len(edge.link_faces) == 2
        ]
        edges_to_rotate = random_edge_subset(internal_edges, p.rotate_edge_ratio)
        stats["rotated_edges"] = rotate_edges_robust(bm, edges_to_rotate)
        triangulate_all(bm)

    stats["merged_verts"] = cleanup_mesh_light(bm, p.merge_distance)
    stats["all_triangles"] = is_all_triangle_mesh(bm)
    return stats


def apply_dirty_topology_to_mesh(
    mesh: bpy.types.Mesh,
    params: DirtyParams | None = None,
) -> dict[str, int | bool | float]:
    """Apply dirty topology in-place to a Mesh datablock."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        stats = apply_dirty_topology_to_bmesh(bm, params)
        bm.to_mesh(mesh)
        mesh.update()
        return stats
    finally:
        bm.free()
