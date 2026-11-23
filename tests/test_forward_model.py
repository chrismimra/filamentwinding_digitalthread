import numpy as np
import pytest

from digital_thread import forward_model as fm

# Hand-worked setup: cylinder about +Z (mandrel turns counter-clockwise about +Z),
# camera on the +X axis 700 mm from the axis, looking at -X, image y pointing to -Z.
R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
T = np.array([0.0, 0.0, 700.0])
AXIS = np.array([0.0, 0.0, 1.0])
POINT = np.array([125.0, 0.0, 0.0])
NORMAL = np.array([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("fibre_deg", "aolp_deg"),
    [(0.0, 0.0), (90.0, 90.0), (45.0, 45.0), (-45.0, -45.0)],
)
def test_aolp_of_fibre_facing_the_camera(fibre_deg, aolp_deg):
    # Fibre along the axis appears vertical in the image, so the light is
    # polarised horizontally (AoLP 0). A hoop fibre gives AoLP 90.
    aolp = fm.aolp_from_fibre(fibre_deg, POINT, NORMAL, AXIS, R, T)
    assert aolp == pytest.approx(aolp_deg, abs=1e-9)


def test_positive_fibre_angle_points_in_rotation_direction():
    # At (125, 0, 0) the surface moves towards +Y when turning about +Z.
    direction = fm.fibre_direction(90.0, NORMAL, AXIS)
    assert direction == pytest.approx([0.0, 1.0, 0.0])


def _random_view(rng, n):
    """Random camera, points in front of it, normals within 70 deg incidence."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = q * np.sign(np.linalg.det(q))
    t = rng.normal(size=3) * 100
    centre = -rot.T @ t
    rays_cam = np.column_stack([rng.uniform(-0.4, 0.4, (n, 2)), np.ones(n)])
    rays = rays_cam @ rot
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    points = centre + rays * rng.uniform(300, 1000, (n, 1))
    normals = -rays + rng.normal(size=(n, 3)) * 0.6
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    keep = (np.sum(-rays * normals, axis=1) > np.cos(np.radians(70))) & (
        np.abs(normals @ axis) < np.cos(np.radians(20))
    )
    return points[keep], normals[keep], axis, rot, t


def test_inverse_recovers_fibre_angle_over_random_geometry():
    rng = np.random.default_rng(0)
    points, normals, axis, rot, t = _random_view(rng, 2000)
    fibre = rng.uniform(-90, 90, len(points))
    aolp = fm.aolp_from_fibre(fibre, points, normals, axis, rot, t)
    recovered = fm.fibre_from_aolp(aolp, points, normals, axis, rot, t)
    assert len(points) > 500
    assert recovered == pytest.approx(fibre, abs=1e-6)


def test_angles_are_wrapped_to_half_open_range():
    assert fm.wrap_angle(np.array([-90.0, 90.0, 135.0, -135.0, 270.0])) == (
        pytest.approx([90.0, 90.0, -45.0, 45.0, 90.0])
    )


def test_polariser_intensities_follow_malus_law():
    # S0 = 1000, DoLP = 0.3, AoLP = 30 deg, worked by hand.
    i0, i45, i90, i135 = (
        fm.polariser_intensity(1000.0, 0.3, 30.0, angle) for angle in (0, 45, 90, 135)
    )
    assert (i0, i90) == pytest.approx((575.0, 425.0))
    assert (i45, i135) == pytest.approx((629.904, 370.096), abs=1e-3)


def test_s0_is_lambertian_with_light_at_camera():
    assert fm.s0(albedo=0.8, cos_incidence=0.5, light=3000.0) == pytest.approx(1200.0)
    assert fm.s0(albedo=0.8, cos_incidence=-0.2, light=3000.0) == 0.0
