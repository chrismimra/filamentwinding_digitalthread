"""Conversion: raw frame + polariser pattern -> polarisation maps.

Expected values come from hand-built raw frames (Malus' law, written out here)
and from the committed ground truth of the synthetic cylinder.
"""

from pathlib import Path

import numpy as np
import pytest

from digital_thread import conversion, dataset, forward_model

PATTERN = (90, 45, 135, 0)  # (top-left, top-right, bottom-left, bottom-right)
SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
SYNTHETIC = Path(__file__).parents[1] / "demoData" / "synthetic"


def raw_block(s0, dolp, aolp_deg, pattern=PATTERN, shape=(2, 3)):
    """Raw frame of uniform light: I = S0/2 (1 + DoLP cos 2(aolp - polariser))."""
    h, w = shape
    raw = np.empty((2 * h, 2 * w))
    for (r, c), polariser in zip(SLOTS, pattern, strict=True):
        raw[r::2, c::2] = (
            0.5 * s0 * (1 + dolp * np.cos(np.radians(2 * (aolp_deg - polariser))))
        )
    return raw.astype(np.uint16) if np.allclose(raw, np.rint(raw)) else raw


@pytest.mark.parametrize("aolp", [0.0, 30.0, -60.0, 90.0])
def test_uniform_light_gives_its_aolp_dolp_and_s0(aolp):
    maps = conversion.convert(raw_block(2000.0, 0.3, aolp), PATTERN)
    assert maps.s0.shape == (2, 3)
    assert maps.s0 == pytest.approx(2000.0, rel=1e-6)
    assert maps.dolp == pytest.approx(0.3, abs=1e-6)
    assert maps.aolp == pytest.approx(aolp, abs=1e-4)  # 90 stays 90, not -90


def test_aolp_lies_in_the_half_open_interval():
    maps = conversion.convert(raw_block(2000.0, 0.3, -90.0), PATTERN)
    assert maps.aolp == pytest.approx(90.0, abs=1e-4)


def test_pattern_order_is_read_from_the_pattern():
    light = (1800.0, 0.25, 30.0)
    ours = conversion.convert(raw_block(*light), PATTERN)
    other = (0, 45, 135, 90)
    theirs = conversion.convert(raw_block(*light, pattern=other), other)
    for name in ("s0", "dolp", "aolp"):
        np.testing.assert_allclose(getattr(theirs, name), getattr(ours, name))
    assert theirs.aolp == pytest.approx(30.0, abs=1e-4)


def test_saturated_channel_gives_nan_only_in_its_super_pixel():
    raw = raw_block(2000.0, 0.3, 30.0)
    raw[2, 3] = 4095  # super-pixel (1, 1)
    maps = conversion.convert(raw, PATTERN)
    nan = np.isnan(maps.aolp)
    assert nan[1, 1] and np.isnan(maps.dolp[1, 1])
    assert nan.sum() == 1 and np.isnan(maps.dolp).sum() == 1
    assert np.isfinite(maps.s0).all()


def test_dark_super_pixel_gives_nan_without_warning():
    raw = np.zeros((4, 4), dtype=np.uint16)
    with np.errstate(all="raise"):
        maps = conversion.convert(raw, PATTERN)
    assert np.isnan(maps.dolp).all() and np.isnan(maps.aolp).all()


def test_maps_are_float32():
    maps = conversion.convert(raw_block(2000.0, 0.3, 30.0), PATTERN)
    assert {maps.s0.dtype, maps.dolp.dtype, maps.aolp.dtype} == {np.dtype(np.float32)}


def test_synthetic_cylinder_new_tow_matches_the_ground_truth():
    run = dataset.read_run(SYNTHETIC / "cylinder" / "run")
    cal = dataset.read_calibration(run.calibration_folder)
    geometry = np.load(run.folder / "truth_geometry.npz")
    truth = np.load(run.folder / "rev002_truth.npz")
    new_tow = truth["revolution"] == 2

    maps = conversion.convert(dataset.read_frame(run, "rev002"), run.polar_pattern)
    expected = forward_model.aolp_from_fibre(
        truth["fibre_deg"][new_tow],
        geometry["point"][new_tow],
        geometry["normal"][new_tow],
        cal.axis_dir,
        cal.R,
        cal.t,
    )
    assert np.isfinite(maps.aolp[new_tow]).mean() > 0.99  # no hidden NaN
    error = (maps.aolp[new_tow] - expected + 90) % 180 - 90
    assert np.nanmean(maps.dolp[new_tow]) == pytest.approx(0.3, abs=0.02)
    # Signed median = bias; single pixels scatter by about 1 deg from shot noise.
    assert abs(np.nanmedian(error)) < 0.2
