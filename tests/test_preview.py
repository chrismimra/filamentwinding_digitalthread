"""Preview checks (slice 1) on small hand-made frames."""

import numpy as np

from digital_thread import preview


def test_saturation_marks_super_pixels_with_any_channel_at_full_scale():
    raw = np.full((4, 6), 1000, dtype=np.uint16)
    raw[0, 1] = 4095  # super-pixel (0, 0)
    raw[3, 5] = 4095  # super-pixel (1, 2)
    raw[2, 2] = 4094  # just below
    mask, count = preview.saturation(raw)
    expected = np.zeros((2, 3), dtype=bool)
    expected[0, 0] = expected[1, 2] = True
    np.testing.assert_array_equal(mask, expected)
    assert count == 2


def test_focus_value_is_higher_for_a_sharp_image():
    sharp = np.zeros((64, 64))
    sharp[:, 32:] = 1000.0  # one sharp edge
    ramp = np.clip((np.arange(64) - 16) / 32, 0, 1) * 1000.0
    blurred = np.tile(ramp, (64, 1))  # the same edge spread over 32 pixels
    value_sharp, edges = preview.focus(sharp)
    value_blurred, _ = preview.focus(blurred)
    assert value_sharp > 10 * value_blurred
    assert (
        edges.shape == sharp.shape and edges[:, 31:33].all() and not edges[:, :28].any()
    )
