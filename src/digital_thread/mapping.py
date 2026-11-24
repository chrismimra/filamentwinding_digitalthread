# Mapping (section 5). Production copy of the notebook cell, kept identical
# by the reference test.

import numpy as np

from digital_thread.forward_model import cos_incidence, fibre_from_aolp


def valid_pixels(maps, geo, cal, processing):
    """Valid pixel: inside the crop polygon and on the mandrel (geometry not NaN),
    incidence <= limit, DoLP >= threshold and a finite AoLP."""
    hit = ~np.isnan(geo.point[..., 0])
    valid = np.zeros(hit.shape, dtype=bool)
    point = geo.point[hit].astype(np.float64)
    normal = geo.normal[hit].astype(np.float64)
    steep = cos_incidence(point, normal, cal.R, cal.t) < np.cos(
        np.radians(processing.incidence_limit_deg)
    )
    dolp, aolp = maps.dolp[hit], maps.aolp[hit]
    with np.errstate(invalid="ignore"):  # NaN DoLP compares as False
        polarised = dolp >= processing.dolp_threshold
    valid[hit] = ~steep & polarised & np.isfinite(aolp)
    return valid


def fibre_angles(maps, geo, cal, processing):
    """Fibre angle (deg) per valid pixel by the inverse forward model, NaN elsewhere."""
    valid = valid_pixels(maps, geo, cal, processing)
    fibre = np.full(valid.shape, np.nan, dtype=np.float32)
    fibre[valid] = fibre_from_aolp(
        maps.aolp[valid].astype(np.float64),
        geo.point[valid].astype(np.float64),
        geo.normal[valid].astype(np.float64),
        cal.axis_dir,
        cal.R,
        cal.t,
    )
    return fibre


def axis_frame(axis_dir):
    """Unit axis a and the directions e1, e2 of body angle 0 and 90 deg about it.
    Body angle zero is the STEP X direction projected normal to the axis (Y if
    the axis is along X), counter-clockwise about a, as in the touch-down model."""
    a = np.asarray(axis_dir, dtype=float)
    a = a / np.linalg.norm(a)
    zero = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = zero - (zero @ a) * a
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def flat_coordinates(points, axis_point, axis_dir, r_ref, phi_centre_deg):
    """Flattened coordinates (mm) of surface points: axial position along the axis
    and arc length r_ref * phi, with the body angle phi within +-180 deg of the
    view centre."""
    a, e1, e2 = axis_frame(axis_dir)
    rel = np.asarray(points, dtype=float) - axis_point
    phi = np.degrees(np.arctan2(rel @ e2, rel @ e1))
    phi = phi_centre_deg + (phi - phi_centre_deg + 180.0) % 360.0 - 180.0
    return rel @ a, r_ref * np.radians(phi)


def flatten(per_pixel, geo):
    """Flattened image of a per-pixel map: each grid cell takes the value of the
    super-pixel that sees its surface point (NaN where hidden or outside)."""
    image = np.full(geo.flat_pixel.shape, np.nan, dtype=np.float32)
    seen = geo.flat_pixel >= 0
    image[seen] = np.asarray(per_pixel).reshape(-1)[geo.flat_pixel[seen]]
    return image
