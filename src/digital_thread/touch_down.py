"""Touch-down model: in which revolution each planned-path point is laid.

Copied from the notebook, section 6 "Touch-down model" code cell. The notebook
is the reference; keep both identical (checked by the reference test).
"""

import numpy as np


def body_angle(points, axis_point, axis_dir):
    """Unwrapped angle (deg) of each path point about the axis, counter-clockwise
    about axis_dir (right-hand rule), in path order. Zero is the STEP X
    direction projected normal to the axis (Y if the axis is along X)."""
    a = np.asarray(axis_dir, dtype=float)
    a = a / np.linalg.norm(a)
    zero = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = zero - (zero @ a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    rel = np.asarray(points, dtype=float) - axis_point
    return np.degrees(np.unwrap(np.arctan2(rel @ e2, rel @ e1)))


def mandrel_angle(points, axis_point, axis_dir, touch_down_deg, pin_phase_deg):
    """Mandrel angle (deg, counted from the last pin pulse before winding) at
    which each point is laid. A point at body angle phi touches down when
    mandrel angle + phi = touch-down angle, so later tow lies at smaller phi.
    The first point is laid within the first revolution."""
    phi = body_angle(points, axis_point, axis_dir)
    first = np.mod(touch_down_deg - phi[0] - pin_phase_deg, 360.0)
    return first + (phi[0] - phi)


def revolution_index(points, axis_point, axis_dir, touch_down_deg, pin_phase_deg):
    """Revolution k in which each point is laid (k = 0 for the first point)."""
    theta = mandrel_angle(points, axis_point, axis_dir, touch_down_deg, pin_phase_deg)
    return np.floor(theta / 360.0).astype(int)
