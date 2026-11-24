"""Flattened image on the synthetic cylinder.

On a cylinder of r = 125 mm about Z (body angle zero at +X), the flattened
coordinates (axial position, arc length) of a surface point are exactly
(z, 125 * phi), and a helical tow is a straight band at its fibre angle.
Expected values come from this analytic cylinder and the committed truth.
"""

import json

import numpy as np
import pytest

from digital_thread import dataset, mapping

HELIX_FIBRE_DEG = -np.degrees(np.arctan2(2 * np.pi * 125, 360 / 3))


def cell_centres(geo):
    gh, gw = geo.flat_pixel.shape
    axial = geo.flat_origin[0] + geo.flat_spacing * np.arange(gw)
    arc = geo.flat_origin[1] + geo.flat_spacing * np.arange(gh)
    return np.meshgrid(axial, arc)


def test_reference_radius_is_the_cylinder_radius(cylinder_geometry):
    _, _, geo = cylinder_geometry
    assert geo.r_ref == pytest.approx(125, abs=0.05)
    assert geo.flat_spacing == 0.1


def test_each_grid_cell_shows_the_pixel_that_sees_its_surface_point(cylinder_geometry):
    _, _, geo = cylinder_geometry
    axial, arc = cell_centres(geo)
    seen = geo.flat_pixel >= 0
    assert seen.mean() > 0.9  # the crop is a patch of the barrel
    phi = arc[seen] / 125
    expected = np.column_stack([125 * np.cos(phi), 125 * np.sin(phi), axial[seen]])
    actual = geo.point.reshape(-1, 3)[geo.flat_pixel[seen]]
    # A super-pixel covers about 0.25 mm of surface: nearest pixel within 0.2 mm.
    assert np.linalg.norm(actual - expected, axis=1).max() < 0.2


def test_flat_coordinates_of_a_cylinder_point(cylinder_geometry):
    _, cal, geo = cylinder_geometry
    phi = np.radians([-10.0, 0.0, 15.0])
    points = np.column_stack([125 * np.cos(phi), 125 * np.sin(phi), [-50, 0, 120]])
    axial, arc = mapping.flat_coordinates(
        points, cal.axis_point, cal.axis_dir, geo.r_ref, geo.phi_centre_deg
    )
    assert axial == pytest.approx([-50, 0, 120], abs=1e-9)
    assert arc == pytest.approx(geo.r_ref * phi, abs=1e-9)


def test_each_tow_is_a_straight_band_at_its_fibre_angle(cylinder_geometry):
    run, _, geo = cylinder_geometry
    truth = json.loads((run.folder / "truth.json").read_text(encoding="utf-8"))
    revolution = np.load(run.folder / "rev002_truth.npz")["revolution"]
    flat_revolution = mapping.flatten(revolution.astype(float), geo)
    axial, arc = cell_centres(geo)
    for r in truth["revolutions"]:
        band = flat_revolution == r["revolution"]
        assert band.sum() > 10000
        xy = np.column_stack([axial[band], arc[band]])
        centre = xy.mean(axis=0)
        direction = np.linalg.svd(xy - centre, full_matrices=False)[2][0]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        angle = (angle + 90) % 180 - 90
        assert angle == pytest.approx(HELIX_FIBRE_DEG + r["angle_error_deg"], abs=0.1)
        x, y, z = r["crossing_point"]
        crossing = np.array([z, 125 * np.arctan2(y, x)])
        d = crossing - centre
        offset = abs(direction[0] * d[1] - direction[1] * d[0])  # distance to line
        assert offset == pytest.approx(abs(r["offset_mm"]), abs=0.2)


def test_lookup_round_trip(tmp_path, cylinder_geometry):
    _, _, geo = cylinder_geometry
    dataset.write_geometry(tmp_path, geo)
    back = dataset.read_geometry(tmp_path)
    np.testing.assert_array_equal(back.flat_pixel, geo.flat_pixel)
    assert back.flat_origin == pytest.approx(geo.flat_origin)
    assert (back.r_ref, back.phi_centre_deg, back.flat_spacing) == pytest.approx(
        (geo.r_ref, geo.phi_centre_deg, geo.flat_spacing)
    )


def test_body_angle_is_wrapped_around_the_view_centre():
    # View centre at 170 deg: a point at -175 deg lies 15 deg further on, at 185 deg.
    point = np.array(
        [[100 * np.cos(np.radians(-175)), 100 * np.sin(np.radians(-175)), 7]]
    )
    axial, arc = mapping.flat_coordinates(point, np.zeros(3), [0, 0, 1.0], 100, 170)
    assert axial == pytest.approx([7])
    assert arc == pytest.approx([100 * np.radians(185)])
