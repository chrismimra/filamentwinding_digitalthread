"""Calibration (slice 2) on the synthetic calibration images.

Expected values come from `calibration_images/truth.json`: the intrinsics and
camera pose of the banana calibration bundle the images were rendered with.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from digital_thread import calibration, dataset, geometry

IMAGES = Path(__file__).parents[1] / "demoData" / "synthetic" / "calibration_images"


@pytest.fixture(scope="module")
def truth():
    return json.loads((IMAGES / "truth.json").read_text(encoding="utf-8"))


def _images(prefix):
    return [dataset.read_image(f) for f in sorted(IMAGES.glob(f"{prefix}_*.png"))]


def test_one_frontal_image_gives_the_distortion(truth):
    # The baseline: the board facing the camera. The straight tile edges give
    # k1 and k2; the focal length is the lens's nominal value.
    K = np.array(truth["K"])
    result = calibration.intrinsic(_images("board")[:1], focal_px=K[0, 0])
    assert result.used == 1 and result.focal_from_lens and result.centre_fixed
    np.testing.assert_allclose(result.K, K)
    assert result.distortion[0] == pytest.approx(truth["distortion"][0], abs=0.01)
    assert result.rms_px < 0.5


def test_tilted_images_fit_the_focal_length_and_centre(truth):
    K = np.array(truth["K"])
    nominal = 1.05 * K[0, 0]  # the lens is never exactly its nominal value
    result = calibration.intrinsic(_images("board"), focal_px=nominal)
    assert result.used == 7
    assert not (result.focal_from_lens or result.centre_fixed)
    np.testing.assert_allclose(np.diag(result.K)[:2], np.diag(K)[:2], rtol=0.005)
    np.testing.assert_allclose(result.K[:2, 2], K[:2, 2], atol=3.0)
    assert result.distortion[0] == pytest.approx(truth["distortion"][0], abs=0.01)
    assert result.rms_px < 0.5


def test_a_board_partly_outside_the_image_counts():
    counts = [len(calibration.detect_board(image)[1]) for image in _images("board")]
    assert min(counts) >= 16 and min(counts) < 54  # some views show part of it


def test_too_little_of_the_board_does_not_count():
    board = _images("board")[0]
    h, w = board.shape
    corner = np.ascontiguousarray(board[: h // 4, : w // 4])  # about 2 x 2 tiles
    assert calibration.detect_board(corner) is None
    with pytest.raises(ValueError, match="board found in none"):
        calibration.intrinsic([corner])


@pytest.fixture(scope="module")
def banana(truth):
    cal = dataset.read_calibration(IMAGES / truth["calibration"])
    ids, centres = dataset.read_markers(IMAGES / truth["markers"]["table"])
    return cal, ids, centres


def _angle_deg(r1, r2):
    cos = (np.trace(r1.T @ r2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def test_marker_corners_lie_on_the_mesh_with_their_size(banana):
    cal, ids, centres = banana
    winding = np.isin(ids, [8, 9, 10, 19, 20, 21])  # on the winding surface
    corners = calibration.marker_corners(cal, centres[winding], 30.0)
    on_mesh, _ = geometry.closest_on_mesh(cal, corners.reshape(-1, 3))
    np.testing.assert_allclose(on_mesh, corners.reshape(-1, 3), atol=0.1)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
    np.testing.assert_allclose(sides, 30.0, atol=0.5)


def test_marker_top_edge_follows_the_orientation_rule(banana):
    cal, _, _ = banana
    v, n = cal.mesh_vertices, cal.mesh_normals
    end = (n[:, 1] > 0.99) & (v[:, 1] > 899.9)  # rod end face, 32 mm across
    axial = np.median(v[end], axis=0)
    bottom = np.array([0.0, 300.0, -200.0])  # winding surface facing -Z
    corners = calibration.marker_corners(cal, np.array([bottom, axial]), 10.0)
    tl, tr, br, bl = corners.transpose(1, 0, 2)
    up = (tl + tr - br - bl) / 2
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    np.testing.assert_allclose(up[0], [0, 1, 0], atol=0.01)  # along +Y, the axis
    np.testing.assert_allclose(up[1], [0, 0, 1], atol=0.05)  # axial face: +Z
    right = (tr[0] - tl[0]) / np.linalg.norm(tr[0] - tl[0])
    # Seen from outside, right = up x normal (normal -Z here).
    np.testing.assert_allclose(right, np.cross([0, 1, 0], [0, 0, -1]), atol=0.01)


def test_pose_recovered_from_the_marker_images(banana, truth):
    cal, ids, centres = banana
    corners = calibration.marker_corners(cal, centres, 30.0)
    pose = calibration.position(_images("markers"), ids, corners, cal.K, cal.distortion)
    assert pose.used == 5
    assert set(pose.marker_errors_px) == {9, 10, 19, 20, 21}  # 8 just outside
    centre = -np.array(truth["R"]).T @ np.array(truth["t"])
    np.testing.assert_allclose(pose.centre, centre, atol=1.0)
    assert _angle_deg(pose.R, np.array(truth["R"])) < 0.1
    assert max(pose.marker_errors_px.values()) < 1.0
    assert calibration.position_mismatch(pose, centre + [0, 0, 30]) is None
    assert calibration.position_mismatch(pose, centre + [0, 100, 0]) > 50


def test_wrong_marker_position_shows_the_largest_error(banana):
    cal, ids, centres = banana
    wrong = centres.copy()
    wrong[ids == 10] += [0, 10, 0]
    corners = calibration.marker_corners(cal, wrong, 30.0)
    pose = calibration.position(_images("markers"), ids, corners, cal.K, cal.distortion)
    errors = pose.marker_errors_px
    assert max(errors, key=errors.get) == 10 and errors[10] > 2.0


def test_body_angle_facing_the_camera(banana):
    cal, _, _ = banana  # camera on -Z, axis +Y: body angle 90 deg
    assert calibration.body_angle_facing(cal) == pytest.approx(90.0, abs=0.01)


def test_markers_far_from_the_mesh_are_reported(banana):
    cal, ids, centres = banana
    centres = centres.copy()
    centres[ids == 5, 2] = 112.5  # a sign typo: beyond the banana
    _, _, distance = calibration.surface_points(cal, centres)
    off = ids[distance > calibration.MAX_OFF_MESH_MM]
    assert sorted(off) == [5]


def test_bundle_keeps_the_marker_table(banana, tmp_path, truth):
    table = IMAGES / truth["markers"]["table"]
    cal = dataclasses.replace(
        banana[0], marker_size_mm=30.0, intrinsic_rms_px=0.2, position_rms_px=0.3
    )
    calibration.save_bundle(tmp_path, cal, table)
    back = dataset.read_calibration(tmp_path)
    assert (back.marker_size_mm, back.intrinsic_rms_px, back.position_rms_px) == (
        30.0,
        0.2,
        0.3,
    )
    assert (tmp_path / dataset.MARKER_TABLE_FILE).read_bytes() == table.read_bytes()


def test_live_board_check_finds_the_board_only_where_it_is():
    board = _images("board")[0]
    corners = calibration.find_board(board)
    assert corners is not None and len(corners) >= 16
    h, w = board.shape
    assert (
        (corners >= 0).all() and (corners[:, 0] < w).all() and (corners[:, 1] < h).all()
    )
    assert calibration.find_board(_images("markers")[0]) is None
