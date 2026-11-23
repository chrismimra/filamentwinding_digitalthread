import numpy as np
import pytest

from digital_thread import touch_down

AXIS_POINT = np.zeros(3)
AXIS_DIR = np.array([0.0, 0.0, 1.0])


def on_circle(body_angles_deg):
    """Path points at the given body angles about +Z, advancing along Z."""
    a = np.radians(body_angles_deg)
    return np.column_stack([125 * np.cos(a), 125 * np.sin(a), np.arange(len(a)) * 10.0])


# The mandrel turns counter-clockwise about +Z, so later tow lies at smaller
# body angles. Hand-worked: with touch-down at 180 deg and the pin at 0 deg the
# first point (body angle 180) is laid at mandrel angle 0, the start of revolution 0.
PATH_DEG = [180, 90, 0, -90, -180, -270]


def test_revolution_counts_one_per_full_turn_from_the_first_point():
    k = touch_down.revolution_index(
        on_circle(PATH_DEG), AXIS_POINT, AXIS_DIR, touch_down_deg=180, pin_phase_deg=0
    )
    assert k.tolist() == [0, 0, 0, 0, 1, 1]


def test_pin_phase_moves_the_revolution_boundary():
    # Pin at 90 deg: the first point is laid 270 deg after the last pulse.
    k = touch_down.revolution_index(
        on_circle(PATH_DEG), AXIS_POINT, AXIS_DIR, touch_down_deg=180, pin_phase_deg=90
    )
    assert k.tolist() == [0, 1, 1, 1, 1, 2]


def test_touch_down_angle_moves_the_revolution_boundary():
    # Touch-down at 270 deg: the first point is laid at mandrel angle 90.
    k = touch_down.revolution_index(
        on_circle(PATH_DEG), AXIS_POINT, AXIS_DIR, touch_down_deg=270, pin_phase_deg=0
    )
    assert k.tolist() == [0, 0, 0, 1, 1, 1]


def test_mandrel_angle_when_laid():
    theta = touch_down.mandrel_angle(
        on_circle(PATH_DEG), AXIS_POINT, AXIS_DIR, touch_down_deg=180, pin_phase_deg=0
    )
    assert theta == pytest.approx([0, 90, 180, 270, 360, 450])
