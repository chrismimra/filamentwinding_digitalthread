# Tow matching (section 6). Production copy of the notebook cell, kept identical
# by the reference test.

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from digital_thread.forward_model import cos_incidence, fibre_frame, wrap_angle
from digital_thread.mapping import flat_coordinates, flatten

SAME_POINT_MM = 1.0  # the pixel that sees the crossing point lies this close to it
MAX_OFF_PLANE_MM = 2.0  # a revolution that never reaches the plane has no crossing
MIN_CHANGE_DEG = 5.0  # new tow over an older one at another angle
MIN_AREA_MM2 = 40.0  # about one tow width squared; smaller changes are noise
WINDOW_ALONG_MM = 15.0  # measured tow used for the centre line, along the tow
WINDOW_ACROSS_MM = 10.0  # and across it, around the crossing point
MIN_CELLS = 100  # 1 mm2 of measured tow at least, else no measurement


@dataclass
class Crossing:
    """Crossing point of a revolution's planned path (plane through camera centre
    and mandrel axis, camera side), its planned fibre angle and visibility."""

    index: int  # index into the planned path, -1 if the revolution never crosses
    point: np.ndarray  # mm
    normal: np.ndarray
    tangent: np.ndarray  # planned direction of travel
    planned_deg: float
    visible: bool


def crossing(planned, k, cal, geo, processing):
    """Crossing point of revolution k. Visible if a pixel inside the crop polygon
    sees it and its incidence is within the limit."""
    axis = cal.axis_dir / np.linalg.norm(cal.axis_dir)
    centre = -cal.R.T @ cal.t
    towards_camera = centre - cal.axis_point
    towards_camera -= (towards_camera @ axis) * axis
    plane_normal = np.cross(axis, towards_camera)
    plane_normal /= np.linalg.norm(plane_normal)

    rel = planned.point - cal.axis_point
    facing = np.einsum("ij,ij->i", planned.point - centre, planned.normal) < 0
    candidate = (planned.revolution == k) & facing & (rel @ towards_camera > 0)
    off_plane = np.where(candidate, np.abs(rel @ plane_normal), np.inf)
    i = int(np.argmin(off_plane))
    if not off_plane[i] <= MAX_OFF_PLANE_MM:
        nan = np.full(3, np.nan)
        return Crossing(-1, nan, nan, nan, np.nan, False)

    p, n = planned.point[i], planned.normal[i]
    t = planned.point[min(i + 1, len(rel) - 1)] - planned.point[max(i - 1, 0)]
    t /= np.linalg.norm(t)
    e_a, e_r = fibre_frame(n, cal.axis_dir)
    planned_deg = float(wrap_angle(np.degrees(np.arctan2(t @ e_r, t @ e_a))))

    with np.errstate(invalid="ignore"):
        seen = np.nanmin(np.linalg.norm(geo.point - p, axis=-1)) < SAME_POINT_MM
    steep = cos_incidence(p, n, cal.R, cal.t) < np.cos(
        np.radians(processing.incidence_limit_deg)
    )
    return Crossing(i, p, n, t, planned_deg, bool(seen and not steep))


def measured_tow(fibre_before, fibre_after, geo):
    """Measured tow in the flattened image: cells valid after that were invalid
    before or whose fibre angle changed by more than MIN_CHANGE_DEG, in connected
    regions of at least MIN_AREA_MM2."""
    valid_before, valid_after = np.isfinite(fibre_before), np.isfinite(fibre_after)
    with np.errstate(invalid="ignore"):
        turned = np.abs(wrap_angle(fibre_after - fibre_before)) > MIN_CHANGE_DEG
    changed = valid_after & (~valid_before | turned)
    flat = flatten(changed.astype(np.float32), geo) == 1
    labels, _ = ndimage.label(flat)
    area = np.bincount(labels.ravel()) * geo.flat_spacing**2
    keep = area >= MIN_AREA_MM2
    keep[0] = False  # background
    return keep[labels]


def match_revolution(tow_cells, cross, cal, geo):
    """Delta position (mm, along n x t) and Delta angle (deg) of the measured tow
    at the crossing point, from a centre line fitted in the flattened image.

    Coordinates are relative to the crossing point; the circumferential one is
    rescaled from r_ref to the radius at the crossing point. Returns
    (delta_position_mm, delta_angle_deg, cells used).
    """
    axis = cal.axis_dir / np.linalg.norm(cal.axis_dir)
    axial_c, arc_c = flat_coordinates(
        cross.point[None], cal.axis_point, axis, geo.r_ref, geo.phi_centre_deg
    )
    rel = cross.point - cal.axis_point
    scale = np.linalg.norm(rel - (rel @ axis) * axis) / geo.r_ref
    rows, cols = np.nonzero(tow_cells)
    xy = np.column_stack(
        [
            geo.flat_origin[0] + geo.flat_spacing * cols - axial_c[0],
            (geo.flat_origin[1] + geo.flat_spacing * rows - arc_c[0]) * scale,
        ]
    )
    # Planned direction of travel and side n x t in (axial, arc) coordinates.
    e_a, e_r = fibre_frame(cross.normal, cal.axis_dir)
    d = np.array([cross.tangent @ e_a, cross.tangent @ e_r])
    d /= np.linalg.norm(d)
    near = (np.abs(xy @ d) <= WINDOW_ALONG_MM) & (
        np.abs(xy @ np.array([d[1], -d[0]])) <= WINDOW_ACROSS_MM
    )
    if near.sum() < MIN_CELLS:
        return np.nan, np.nan, int(near.sum())
    pts = xy[near]
    centre = pts.mean(axis=0)
    u = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    u = u if u @ d > 0 else -u
    delta_angle = wrap_angle(np.degrees(np.arctan2(u[1], u[0])) - cross.planned_deg)
    delta_position = centre @ np.array([u[1], -u[0]])  # side of the measured line
    return float(delta_position), float(delta_angle), int(near.sum())


def tow_table(run, cal, geo, planned, fibre_maps):
    """One row per revolution: crossing point, planned angle and the measured
    Delta position / Delta angle (NaN if not visible or no tow found).
    fibre_maps: fibre angle per pixel for each frame name of the run.

    The frame log starts with the reference frame; the revolution index comes
    from each frame's name (set from the timestamps in Data capturing), so a
    missed revolution leaves a gap instead of shifting the numbering. The
    measured tow is the change from the previous frame in the log."""
    rows = []
    names = [f.name for f in run.frames]
    for i in range(1, len(names)):
        name, before = names[i], names[i - 1]
        k = int(name.removeprefix("rev"))
        cross = crossing(planned, k, cal, geo, run.processing)
        delta_position = delta_angle = np.nan
        cells = 0
        if cross.visible:
            tow = measured_tow(fibre_maps[before], fibre_maps[name], geo)
            delta_position, delta_angle, cells = match_revolution(tow, cross, cal, geo)
        rows.append(
            {
                "revolution": k,
                "visible": cross.visible,
                "x_mm": float(cross.point[0]),
                "y_mm": float(cross.point[1]),
                "z_mm": float(cross.point[2]),
                "planned_angle_deg": cross.planned_deg,
                "delta_position_mm": delta_position,
                "delta_angle_deg": delta_angle,
                "tow_cells": cells,
            }
        )
    return rows
