"""Run geometry (Mapping, production code): which surface point each super-pixel sees.

Computed once per run, since the camera does not move. Per super-pixel: the
pixel ray (intrinsics, distortion inverted, so only pixel positions change),
the nearest hit on the mandrel mesh and the normal interpolated from the CAD
vertex normals. NaN outside the crop polygon and where the ray misses.
It also holds the flattening lookup: which super-pixel shows each cell of
the flattened image, and the planned path resampled on the mesh (Tow matching).
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from matplotlib.path import Path as Polygon

from digital_thread import touch_down
from digital_thread.dataset import Calibration, RunGeometry
from digital_thread.mapping import axis_frame, flat_coordinates

FLAT_SPACING_MM = 0.1
HIDDEN_TOLERANCE_MM = 1.0  # a hidden surface point lies far further from the seen one
PATH_STEP_MM = 1.0


@dataclass
class PlannedPath:
    """Planned path resampled on the mesh, with outward normals and the revolution
    of each point by the touch-down model."""

    point: np.ndarray  # (n, 3) mm
    normal: np.ndarray  # (n, 3)
    revolution: np.ndarray  # (n,) int


def camera_centre(cal: Calibration) -> np.ndarray:
    return -cal.R.T @ cal.t


def crop_mask(cal: Calibration) -> np.ndarray:
    """True for super-pixels inside the crop polygon (all if there is none)."""
    w, h = cal.image_size
    if cal.crop_polygon is None:
        return np.ones((h, w), dtype=bool)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    inside = Polygon(cal.crop_polygon).contains_points(
        np.column_stack([u.ravel(), v.ravel()])
    )
    return inside.reshape(h, w)


def _distort(x, y, distortion):
    """OpenCV distortion model (k1, k2, p1, p2, k3) on normalised coordinates."""
    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def project(cal: Calibration, points):
    """Super-pixel coordinates (u, v) of world points, with distortion."""
    cam = points @ cal.R.T + cal.t
    xd, yd = _distort(cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2], cal.distortion)
    return cal.K[0, 0] * xd + cal.K[0, 2], cal.K[1, 1] * yd + cal.K[1, 2]


def pixel_rays(cal: Calibration) -> np.ndarray:
    """Unit world ray per super-pixel centre, shape (h, w, 3)."""
    w, h = cal.image_size
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    xd = (u - cal.K[0, 2]) / cal.K[0, 0]
    yd = (v - cal.K[1, 2]) / cal.K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(50):  # fixed-point inversion of the distortion
        fx, fy = _distort(x, y, cal.distortion)
        x, y = x - (fx - xd), y - (fy - yd)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ cal.R  # R^T per pixel
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _scene(cal: Calibration):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(cal.mesh_vertices.astype(np.float32)),
        o3d.core.Tensor(cal.mesh_faces.astype(np.uint32)),
    )
    return scene


def cast(cal: Calibration, origins, directions, scene=None):
    """Nearest mesh hit and interpolated unit normal per ray, NaN on a miss.

    Open3D (float32) finds the triangle; the hit is recomputed in float64.
    """
    scene = scene or _scene(cal)
    origins = np.broadcast_to(origins, directions.shape)
    query = np.hstack([origins, directions]).astype(np.float32)
    ids = scene.cast_rays(o3d.core.Tensor(query))["primitive_ids"].numpy()
    hit = ids != scene.INVALID_ID
    tri = cal.mesh_faces[ids[hit].astype(np.int64)]
    v = cal.mesh_vertices
    v0, e1, e2 = v[tri[:, 0]], v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]
    d, s = directions[hit], origins[hit] - v0
    p, q = np.cross(d, e2), np.cross(s, e1)
    inv = 1.0 / np.einsum("ij,ij->i", e1, p)
    b1 = np.einsum("ij,ij->i", s, p) * inv  # barycentric (Moeller-Trumbore)
    b2 = np.einsum("ij,ij->i", d, q) * inv
    dist = np.einsum("ij,ij->i", e2, q) * inv
    n = cal.mesh_normals
    normal = (
        (1 - b1 - b2)[:, None] * n[tri[:, 0]]
        + b1[:, None] * n[tri[:, 1]]
        + b2[:, None] * n[tri[:, 2]]
    )
    points = np.full(directions.shape, np.nan)
    normals = np.full(directions.shape, np.nan)
    points[hit] = origins[hit] + dist[:, None] * d
    normals[hit] = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    return points, normals


def closest_on_mesh(cal: Calibration, points, scene=None):
    """Closest mesh point and interpolated unit normal for each query point.

    Open3D (float32) finds the triangle; the point is projected onto that
    triangle's plane in float64.
    """
    scene = scene or _scene(cal)
    points = np.asarray(points, dtype=float)
    query = o3d.core.Tensor(points.astype(np.float32))
    ids = scene.compute_closest_points(query)["primitive_ids"].numpy().astype(np.int64)
    tri = cal.mesh_faces[ids]
    v = cal.mesh_vertices
    v0, e1, e2 = v[tri[:, 0]], v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]
    plane_n = np.cross(e1, e2)
    plane_n /= np.linalg.norm(plane_n, axis=1, keepdims=True)
    projected = points - np.einsum("ij,ij->i", points - v0, plane_n)[:, None] * plane_n
    w = projected - v0
    d11, d12, d22 = (
        np.einsum("ij,ij->i", a, b) for a, b in ((e1, e1), (e1, e2), (e2, e2))
    )
    w1, w2 = np.einsum("ij,ij->i", w, e1), np.einsum("ij,ij->i", w, e2)
    det = d11 * d22 - d12 * d12
    b1, b2 = (d22 * w1 - d12 * w2) / det, (d11 * w2 - d12 * w1) / det
    n = cal.mesh_normals
    normal = (
        (1 - b1 - b2)[:, None] * n[tri[:, 0]]
        + b1[:, None] * n[tri[:, 1]]
        + b2[:, None] * n[tri[:, 2]]
    )
    return projected, normal / np.linalg.norm(normal, axis=1, keepdims=True)


def planned_path(
    cal: Calibration, tow_file_points, tow_width_mm: float, step_mm=PATH_STEP_MM
) -> PlannedPath:
    """Tow placement file -> planned path on the mesh.

    Points further than one tow width from the mesh (lead-in/out) are dropped;
    the rest is resampled every `step_mm` along the polyline and projected onto
    the mesh, so a coarse file does not cut below the surface. Revolutions by
    the touch-down model.
    """
    scene = _scene(cal)
    raw = np.asarray(tow_file_points, dtype=float)
    on_mesh, _ = closest_on_mesh(cal, raw, scene)
    raw = raw[np.linalg.norm(on_mesh - raw, axis=1) <= tow_width_mm]
    length = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    )
    s = np.arange(0.0, length[-1], step_mm)
    dense = np.column_stack([np.interp(s, length, raw[:, i]) for i in range(3)])
    point, normal = closest_on_mesh(cal, dense, scene)
    revolution = touch_down.revolution_index(
        point, cal.axis_point, cal.axis_dir, cal.touch_down_deg, cal.pin_phase_deg
    )
    return PlannedPath(point, normal, revolution)


def _view_centre(point):
    """Hit point of the super-pixel nearest the image centre."""
    h, w = point.shape[:2]
    rows, cols = np.nonzero(~np.isnan(point[..., 0]))
    nearest = np.argmin((rows - h / 2) ** 2 + (cols - w / 2) ** 2)
    return point[rows[nearest], cols[nearest]].astype(float)


def flattening_lookup(cal: Calibration, point, spacing=FLAT_SPACING_MM, scene=None):
    """Grid of the flattened image and, per cell, the super-pixel that sees it.

    Each cell (axial position, arc length) casts a ray towards the axis from
    outside the mandrel; the hit is assigned to the super-pixel it projects to
    if that super-pixel's own point is the same surface point (else hidden).
    """
    h, w = point.shape[:2]
    a, e1, e2 = axis_frame(cal.axis_dir)
    centre = _view_centre(point) - cal.axis_point
    r_ref = float(np.linalg.norm(centre - (centre @ a) * a))
    phi_centre = float(np.degrees(np.arctan2(centre @ e2, centre @ e1)))

    hit = ~np.isnan(point[..., 0])
    axial, arc = flat_coordinates(
        point[hit].astype(float), cal.axis_point, a, r_ref, phi_centre
    )
    origin = (
        float(np.floor(axial.min() / spacing) * spacing),
        float(np.floor(arc.min() / spacing) * spacing),
    )
    gw = int(np.ceil((axial.max() - origin[0]) / spacing)) + 1
    gh = int(np.ceil((arc.max() - origin[1]) / spacing)) + 1
    cell_axial, cell_arc = np.meshgrid(
        origin[0] + spacing * np.arange(gw), origin[1] + spacing * np.arange(gh)
    )
    phi = (cell_arc / r_ref).reshape(-1, 1)
    radial = np.cos(phi) * e1 + np.sin(phi) * e2
    rel = cal.mesh_vertices - cal.axis_point
    outside = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1).max() + 10.0
    origins = cal.axis_point + cell_axial.reshape(-1, 1) * a + outside * radial
    surface, _ = cast(cal, origins, -radial, scene)

    lookup = np.full(gh * gw, -1, dtype=np.int32)
    ok = np.flatnonzero(~np.isnan(surface[:, 0]))
    u, v = project(cal, surface[ok])
    col, row = np.rint(u).astype(int), np.rint(v).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    cells, pixel = ok[inside], row[inside] * w + col[inside]
    seen = point.reshape(-1, 3)[pixel]
    same = np.linalg.norm(seen - surface[cells], axis=1) < HIDDEN_TOLERANCE_MM
    lookup[cells[same]] = pixel[same]
    return lookup.reshape(gh, gw), origin, r_ref, phi_centre


def compute(cal: Calibration) -> RunGeometry:
    """Geometry of every super-pixel inside the crop polygon, and the flattening lookup."""
    w, h = cal.image_size
    scene = _scene(cal)
    inside = crop_mask(cal)
    points, normals = cast(cal, camera_centre(cal), pixel_rays(cal)[inside], scene)
    point = np.full((h, w, 3), np.nan, dtype=np.float32)
    normal = np.full((h, w, 3), np.nan, dtype=np.float32)
    point[inside], normal[inside] = points, normals
    lookup, origin, r_ref, phi_centre = flattening_lookup(cal, point, scene=scene)
    return RunGeometry(
        point=point,
        normal=normal,
        flat_pixel=lookup,
        flat_origin=origin,
        flat_spacing=FLAT_SPACING_MM,
        r_ref=r_ref,
        phi_centre_deg=phi_centre,
    )
