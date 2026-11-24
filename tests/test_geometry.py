"""Run geometry on the synthetic cylinder, checked against the analytic cylinder.

Expected values come from the STEP geometry (r = 125 mm about Z) and a
hand-written pinhole projection, not from the generator's ray casting.
"""

import numpy as np
import pytest
from matplotlib.path import Path as Polygon

from digital_thread import dataset


@pytest.fixture(scope="module")
def cylinder(cylinder_geometry):
    _, cal, geo = cylinder_geometry
    return cal, geo


def project(points, cal):
    cam = points @ cal.R.T + cal.t
    xn, yn = cam[..., 0] / cam[..., 2], cam[..., 1] / cam[..., 2]
    radial = 1 + cal.distortion[0] * (xn**2 + yn**2)
    return (
        cal.K[0, 0] * xn * radial + cal.K[0, 2],
        cal.K[1, 1] * yn * radial + cal.K[1, 2],
    )


def inside_crop(cal):
    w, h = cal.image_size
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    inside = Polygon(cal.crop_polygon).contains_points(
        np.column_stack([u.ravel(), v.ravel()])
    )
    return inside.reshape(h, w)


def test_every_pixel_inside_the_crop_hits_the_barrel(cylinder):
    cal, geo = cylinder
    w, h = cal.image_size
    assert geo.point.shape == (h, w, 3) and geo.normal.shape == (h, w, 3)
    inside = inside_crop(cal)
    hit = ~np.isnan(geo.point[..., 0])
    assert hit[inside].all()
    assert not hit[~inside].any()
    assert np.isnan(geo.normal[~inside]).all()


def test_points_lie_on_the_cylinder_and_reproject_to_their_pixel(cylinder):
    cal, geo = cylinder
    hit = ~np.isnan(geo.point[..., 0])
    points = geo.point[hit].astype(float)
    assert np.hypot(points[:, 0], points[:, 1]) == pytest.approx(125, abs=0.05)
    rows, cols = np.nonzero(hit)
    u, v = project(points, cal)
    assert np.abs(u - cols).max() < 0.05
    assert np.abs(v - rows).max() < 0.05


def test_normals_are_radial(cylinder):
    _, geo = cylinder
    hit = ~np.isnan(geo.point[..., 0])
    radial = geo.point[hit] * [1, 1, 0] / 125
    assert np.einsum("ij,ij->i", geo.normal[hit], radial) == pytest.approx(1, abs=1e-3)


def test_geometry_round_trip(tmp_path, cylinder):
    _, geo = cylinder
    dataset.write_geometry(tmp_path, geo)
    back = dataset.read_geometry(tmp_path)
    np.testing.assert_array_equal(back.point, geo.point)
    np.testing.assert_array_equal(back.normal, geo.normal)
