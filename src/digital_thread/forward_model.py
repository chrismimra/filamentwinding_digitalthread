"""Forward model: fibre angle and viewing geometry -> AoLP, DoLP and S0.

Copied from the notebook, section 5 "Forward model" code cell. The notebook is
the reference; keep both identical (checked by the reference test).

Conventions: world vectors have shape (..., 3) in the STEP frame (mm). The
camera pose maps world to camera coordinates, x_cam = R @ X + t, with camera
x to the right, y downwards and z forward. Image angles run from +x towards +y.
The mandrel axis direction is oriented so that the mandrel turns
counter-clockwise about it (right-hand rule). Normals point out of the mandrel.
"""

import numpy as np

DOLP = {"tow": 0.3, "mandrel": 0.05, "background": 0.02}
ALBEDO = {"tow": 0.4, "mandrel": 0.8}


def wrap_angle(angle_deg):
    """Wrap an orientation angle to (-90, 90] degrees."""
    return 90.0 - np.mod(90.0 - np.asarray(angle_deg, dtype=float), 180.0)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fibre_frame(normal, axis_dir):
    """Local surface frame (e_a, e_r): axis direction projected into the
    tangent plane, and the surface direction of mandrel rotation."""
    normal = np.asarray(normal, dtype=float)
    axis_dir = np.asarray(axis_dir, dtype=float)
    e_a = _unit(axis_dir - np.sum(axis_dir * normal, axis=-1, keepdims=True) * normal)
    e_r = np.cross(e_a, normal)
    return e_a, e_r


def fibre_direction(fibre_deg, normal, axis_dir):
    """Unit fibre direction t = cos(a) e_a + sin(a) e_r."""
    e_a, e_r = fibre_frame(normal, axis_dir)
    a = np.radians(np.asarray(fibre_deg, dtype=float))[..., None]
    return np.cos(a) * e_a + np.sin(a) * e_r


def _view_rays(point, rot, t):
    centre = -np.asarray(rot).T @ np.asarray(t, dtype=float)
    return _unit(np.asarray(point, dtype=float) - centre)


def cos_incidence(point, normal, rot, t):
    """Cosine of the angle between surface normal and the direction to the camera."""
    return np.sum(-_view_rays(point, rot, t) * np.asarray(normal), axis=-1)


def aolp_from_fibre(fibre_deg, point, normal, axis_dir, rot, t):
    """AoLP (deg) seen by the camera: light polarised along E = v x t."""
    v = _view_rays(point, rot, t)
    e = np.cross(v, fibre_direction(fibre_deg, normal, axis_dir))
    e_cam = e @ np.asarray(rot).T
    return wrap_angle(np.degrees(np.arctan2(e_cam[..., 1], e_cam[..., 0])))


def fibre_from_aolp(aolp_deg, point, normal, axis_dir, rot, t):
    """Inverse: rebuild E from the AoLP with E perpendicular to the ray, then
    the fibre is perpendicular to E and lies in the surface: t = E x n."""
    rot = np.asarray(rot)
    v_cam = _view_rays(point, rot, t) @ rot.T
    psi = np.radians(np.asarray(aolp_deg, dtype=float))
    ex, ey = np.cos(psi), np.sin(psi)
    ez = -(ex * v_cam[..., 0] + ey * v_cam[..., 1]) / v_cam[..., 2]
    e = np.stack([ex, ey, ez], axis=-1) @ rot
    fibre = np.cross(e, normal)
    e_a, e_r = fibre_frame(normal, axis_dir)
    along_r = np.sum(fibre * e_r, axis=-1)
    along_a = np.sum(fibre * e_a, axis=-1)
    return wrap_angle(np.degrees(np.arctan2(along_r, along_a)))


def s0(albedo, cos_incidence, light):
    """Total intensity, Lambertian with the light at the camera."""
    return light * albedo * np.clip(cos_incidence, 0.0, None)


def polariser_intensity(s0, dolp, aolp_deg, polariser_deg):
    """Intensity behind a linear polariser at polariser_deg (Malus)."""
    angle = np.radians(2.0 * (np.asarray(aolp_deg) - polariser_deg))
    return 0.5 * s0 * (1.0 + dolp * np.cos(angle))
