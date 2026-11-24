"""Mapping on the synthetic cylinder: valid pixels and fibre angles.

Expected values come from the committed ground truth (true fibre angle and
revolution per super-pixel) and the analytic cylinder (radial normals).
"""

import json

import numpy as np
import pytest

from digital_thread import conversion, dataset, mapping


@pytest.fixture(scope="module")
def rev002(cylinder_geometry):
    run, cal, geo = cylinder_geometry
    maps = conversion.convert(dataset.read_frame(run, "rev002"), run.polar_pattern)
    truth = np.load(run.folder / "rev002_truth.npz")
    valid = mapping.valid_pixels(maps, geo, cal, run.processing)
    fibre = mapping.fibre_angles(maps, geo, cal, run.processing)
    return run, cal, geo, maps, truth, valid, fibre


def analytic_incidence_deg(geo, cal):
    """Incidence on the cylinder: radial normal against the line of sight."""
    point = geo.point.astype(float)
    normal = point * [1, 1, 0]
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    to_camera = -cal.R.T @ cal.t - point
    to_camera /= np.linalg.norm(to_camera, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(normal * to_camera, -1), -1, 1)))


def test_new_tow_within_the_limit_gets_the_true_fibre_angle(rev002):
    _, cal, geo, _, truth, valid, fibre = rev002
    new_tow = truth["revolution"] == 2
    seen = new_tow & (analytic_incidence_deg(geo, cal) <= 19.5)
    assert seen.sum() > 1000
    assert valid[seen].mean() > 0.99
    error = (fibre[seen] - truth["fibre_deg"][seen] + 90) % 180 - 90
    # Signed median = bias; single pixels scatter by about 1 deg from shot noise.
    assert abs(np.nanmedian(error)) < 0.2


def test_no_pixel_beyond_the_limits_is_valid(rev002):
    run, cal, geo, maps, _, valid, fibre = rev002
    with np.errstate(invalid="ignore"):
        steep = analytic_incidence_deg(geo, cal) > 20.5
        weak = maps.dolp < run.processing.dolp_threshold
    assert not valid[steep].any()
    assert not valid[weak].any()
    assert np.isnan(fibre[~valid]).all()
    assert np.isfinite(fibre[valid]).all()


def test_bare_mandrel_and_background_are_invalid(rev002):
    _, _, geo, _, truth, valid, _ = rev002
    assert not valid[np.isnan(geo.point[..., 0])].any()
    bare = (truth["revolution"] == -1) & ~np.isnan(geo.point[..., 0])
    # Shot noise lifts a few bare pixels (DoLP 0.05) over the 0.1 threshold.
    assert valid[bare].mean() < 0.001


def test_fibre_angles_round_trip(tmp_path, rev002):
    *_, fibre = rev002
    dataset.write_fibre(tmp_path, "rev002", fibre)
    np.testing.assert_array_equal(dataset.read_fibre(tmp_path, "rev002"), fibre)


def test_flattened_fibre_angle_shows_the_new_tow_as_a_straight_band(rev002):
    run, _, geo, _, truth, _, fibre = rev002
    k2 = json.loads((run.folder / "truth.json").read_text(encoding="utf-8"))[
        "revolutions"
    ][2]
    expected = -np.degrees(np.arctan2(2 * np.pi * 125, 360 / 3)) + k2["angle_error_deg"]
    flat = mapping.flatten(fibre, geo)
    band = (mapping.flatten(truth["revolution"].astype(float), geo) == 2) & np.isfinite(
        flat
    )
    assert band.sum() > 10000
    assert np.median(flat[band]) == pytest.approx(expected, abs=0.2)
    rows, cols = np.nonzero(band)
    xy = np.column_stack([cols, rows]) * geo.flat_spacing  # (axial, arc) in mm
    direction = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)[2][0]
    angle = (np.degrees(np.arctan2(direction[1], direction[0])) + 90) % 180 - 90
    assert angle == pytest.approx(expected, abs=0.1)
