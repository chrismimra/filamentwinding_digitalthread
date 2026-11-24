"""Tests of the committed synthetic datasets, read through the dataset I/O module.

Shape-independent tests run on both mandrels (cylinder and banana). The
cylinder-only tests compare against independent sources: the camera data
sheet (16 mm lens, 3.45 um pixels -> 6.9 um super-pixels), the STEP geometry
(cylinder radius 125 mm about Z) and a hand-written pinhole projection.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Polygon

from digital_thread import dataset, touch_down

BARREL_HALF_LENGTH = 185  # r = 125 mm section of WindingDummieSym.step
SYNTHETIC = Path(__file__).parents[1] / "demoData" / "synthetic"
# Mandrel -> (incidence limit of its run, revolutions whose tow is not visible)
MANDRELS = {"cylinder": (20.0, set()), "banana": (30.0, {1})}


def load(name):
    run = dataset.read_run(SYNTHETIC / name / "run")
    cal = dataset.read_calibration(run.calibration_folder)
    geometry = np.load(run.folder / "truth_geometry.npz")
    return run, cal, geometry


@pytest.fixture(scope="module", params=list(MANDRELS))
def mandrel(request):
    return load(request.param)


@pytest.fixture(scope="module")
def cylinder():
    return load("cylinder")


def project(points, cal):
    """Pinhole projection with radial distortion k1, written independently."""
    cam = points @ cal.R.T + cal.t
    xn, yn = cam[..., 0] / cam[..., 2], cam[..., 1] / cam[..., 2]
    radial = 1 + cal.distortion[0] * (xn**2 + yn**2)
    k = cal.K
    return k[0, 0] * xn * radial + k[0, 2], k[1, 1] * yn * radial + k[1, 2]


def truth(run):
    return json.loads((run.folder / "truth.json").read_text(encoding="utf-8"))


def visible(run):
    """Revolutions whose crossing lies inside the valid view."""
    return [r["revolution"] for r in truth(run)["revolutions"] if r["visible"]]


def mandrel_name(run):
    return run.run_id.removeprefix("synthetic_")


def revolution_truth(run, k):
    return np.load(run.folder / f"rev{k:03d}_truth.npz")


def polariser_images(run, name, mask):
    raw = dataset.read_frame(run, name).astype(float)
    slot = dict(zip(run.polar_pattern, [(0, 0), (0, 1), (1, 0), (1, 1)], strict=True))
    return {angle: raw[r::2, c::2][mask] for angle, (r, c) in slot.items()}


# --- Both mandrels ----------------------------------------------------------


def test_calibration_bundle_describes_the_camera(mandrel):
    _, cal, _ = mandrel
    assert cal.image_size == (2048, 1500)
    assert cal.K[0, 0] == pytest.approx(16 / 0.0069)
    assert cal.distortion[0] == pytest.approx(-0.05)
    assert np.linalg.norm(cal.axis_dir) == pytest.approx(1)
    assert len(cal.mesh_faces) > 1000
    assert cal.mesh_faces.max() < len(cal.mesh_vertices)
    assert cal.mesh_normals.shape == cal.mesh_vertices.shape


def test_camera_is_700_mm_from_the_axis(mandrel):
    _, cal, _ = mandrel
    rel = -cal.R.T @ cal.t - cal.axis_point
    radial = rel - (rel @ cal.axis_dir) * cal.axis_dir
    assert np.linalg.norm(radial) == pytest.approx(700)


def test_run_starts_with_a_reference_frame_and_all_frames_are_12_bit(mandrel):
    run, cal, _ = mandrel
    assert run.frames[0].name == "reference"
    width, height = cal.image_size
    for frame in run.frames:
        assert (run.folder / frame.file).exists()
        raw = dataset.read_frame(run, frame.name)
        assert raw.shape == (2 * height, 2 * width)
        assert raw.dtype == np.uint16
        assert raw.max() <= 4095


def test_mandrel_is_brighter_than_background_in_the_reference_frame(mandrel):
    run, _, geometry = mandrel
    raw = dataset.read_frame(run, "reference").astype(float)
    super_pixel = raw[0::2, 0::2] + raw[0::2, 1::2] + raw[1::2, 0::2] + raw[1::2, 1::2]
    on_mandrel = ~np.isnan(geometry["point"][..., 0])
    assert super_pixel[on_mandrel].mean() > 5 * super_pixel[~on_mandrel].mean()


def test_run_has_reference_and_three_revolutions_one_period_apart(mandrel):
    run, _, _ = mandrel
    assert [f.name for f in run.frames] == ["reference", "rev000", "rev001", "rev002"]
    steps = np.diff([f.timestamp_s for f in run.frames])
    assert steps == pytest.approx(run.nominal_period_s)


def test_ground_truth_errors_are_within_bounds_and_zero_in_revolution_0(mandrel):
    run, _, _ = mandrel
    revs = truth(run)["revolutions"]
    assert [r["revolution"] for r in revs] == [0, 1, 2]
    assert (revs[0]["offset_mm"], revs[0]["angle_error_deg"]) == (0, 0)
    for r in revs:
        assert abs(r["offset_mm"]) <= 2
        assert abs(r["angle_error_deg"]) <= 1


def test_each_visible_revolution_lays_a_tow_on_top(mandrel):
    run, _, _ = mandrel
    assert visible(run)
    for k in visible(run):
        assert (revolution_truth(run, k)["revolution"] == k).sum() > 2000


def test_a_not_visible_revolution_crosses_outside_the_valid_view(mandrel):
    # Valid view: inside the crop polygon and within the incidence limit. The
    # tow of such a revolution may still cut a corner of the view elsewhere.
    run, cal, _ = mandrel
    centre = -cal.R.T @ cal.t
    polygon = Polygon(cal.crop_polygon)
    cos_limit = np.cos(np.radians(run.processing.incidence_limit_deg))
    hidden = [r for r in truth(run)["revolutions"] if not r["visible"]]
    assert {r["revolution"] for r in hidden} == MANDRELS[mandrel_name(run)][1]
    for r in hidden:
        p = np.array(r["crossing_point"])
        nearest = np.argmin(np.linalg.norm(cal.mesh_vertices - p, axis=1))
        normal = cal.mesh_normals[nearest]  # CAD normal at the closest vertex
        to_camera = (centre - p) / np.linalg.norm(centre - p)
        u, v = project(p[None], cal)
        in_view = (
            polygon.contains_point((u[0], v[0])) and normal @ to_camera >= cos_limit
        )
        assert not in_view


def distance_to_polyline(point, polyline):
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    s = np.clip(
        np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1
    )
    return np.linalg.norm(a + s[:, None] * ab - point, axis=1).min()


def test_as_laid_path_is_offset_at_the_crossing_point(mandrel):
    run, _, _ = mandrel
    as_laid = dataset.read_path(run.folder / "as_laid_path.csv")
    for r in truth(run)["revolutions"]:
        crossing = np.array(r["crossing_point"])
        offset = distance_to_polyline(crossing, as_laid)
        assert offset == pytest.approx(abs(r["offset_mm"]), abs=0.05)


def test_bare_mandrel_aolp_does_not_change_between_frames(mandrel):
    # Only a new tow may change the AoLP (measured tow = change between frames).
    run, _, geometry = mandrel
    never_covered = (revolution_truth(run, 2)["revolution"] == -1) & ~np.isnan(
        geometry["point"][..., 0]
    )

    def aolp(name):
        i = polariser_images(run, name, never_covered)
        return np.degrees(0.5 * np.arctan2(i[45] - i[135], i[0] - i[90]))

    change = np.abs((aolp("rev002") - aolp("reference") + 90) % 180 - 90)
    assert np.median(change) < 15  # a redrawn random AoLP would give about 45


def test_crossing_points_lie_in_their_revolution_by_the_touch_down_model(mandrel):
    run, cal, _ = mandrel
    planned = dataset.read_path(run.folder / run.tow_file)
    k = touch_down.revolution_index(
        planned, cal.axis_point, cal.axis_dir, cal.touch_down_deg, cal.pin_phase_deg
    )
    assert sorted(set(k)) == [0, 1, 2]
    for r in truth(run)["revolutions"]:
        nearest = np.argmin(np.linalg.norm(planned - r["crossing_point"], axis=1))
        assert k[nearest] == r["revolution"]


def test_new_tow_shows_the_tow_dolp_in_its_frame(mandrel):
    run, _, _ = mandrel
    for k in visible(run):
        new_tow = revolution_truth(run, k)["revolution"] == k
        i = polariser_images(run, f"rev{k:03d}", new_tow)
        lit = i[0] + i[90] > 0  # grazing silhouette pixels can receive no light
        dolp = np.hypot(i[0] - i[90], i[45] - i[135])[lit] / (i[0] + i[90])[lit]
        assert dolp.mean() == pytest.approx(0.3, abs=0.02)


def test_crop_polygon_encloses_the_visible_crossings(mandrel):
    run, cal, _ = mandrel
    assert cal.crop_polygon is not None and 3 <= len(cal.crop_polygon) <= 8
    crossings = np.array(
        [r["crossing_point"] for r in truth(run)["revolutions"] if r["visible"]]
    )
    u, v = project(crossings, cal)
    assert Polygon(cal.crop_polygon).contains_points(np.column_stack([u, v])).all()


def test_run_records_the_processing_settings(mandrel):
    run, _, _ = mandrel
    limit = MANDRELS[mandrel_name(run)][0]
    assert run.processing == dataset.Processing(0.1, limit)
    assert '"processing"' in (run.folder / "run.json").read_text(encoding="utf-8")


# --- Cylinder only (analytic geometry) --------------------------------------

# Spec: one helical pass of 3 revolutions over z = -180..180 mm on r = 125 mm.
# Later tow lies at smaller body angle (counter-clockwise rotation about +Z),
# so the fibre leans against the rotation direction: negative fibre angle.
HELIX_FIBRE_DEG = -np.degrees(np.arctan2(2 * np.pi * 125, 360 / 3))


def test_cylinder_axis_is_z_through_the_origin(cylinder):
    _, cal, _ = cylinder
    assert cal.axis_dir == pytest.approx([0, 0, 1])
    assert cal.axis_point == pytest.approx([0, 0, 0])


def test_truth_points_lie_on_the_cylinder_and_project_to_their_pixel(cylinder):
    _, cal, geometry = cylinder
    points = geometry["point"]
    assert points.shape == (1500, 2048, 3)
    barrel = np.abs(points[..., 2]) < BARREL_HALF_LENGTH
    assert barrel.sum() > 0.3 * barrel.size
    radius = np.hypot(points[barrel][:, 0], points[barrel][:, 1])
    assert radius == pytest.approx(125, abs=0.05)

    rows, cols = np.nonzero(barrel)
    u, v = project(points[barrel], cal)
    assert np.abs(u - cols).max() < 0.05
    assert np.abs(v - rows).max() < 0.05


def test_truth_normals_are_radial_and_face_the_camera(cylinder):
    _, cal, geometry = cylinder
    points, normals = geometry["point"], geometry["normal"]
    barrel = np.abs(points[..., 2]) < BARREL_HALF_LENGTH
    radial = points[barrel] * [1, 1, 0] / 125
    assert np.einsum("ij,ij->i", normals[barrel], radial) == pytest.approx(1, abs=1e-3)
    centre = -cal.R.T @ cal.t
    to_camera = centre - points[barrel]
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    # Silhouette pixels are grazing (cosine ~ 0); the faceted mesh reaches
    # slightly past the true tangent point, hence the small tolerance.
    assert np.einsum("ij,ij->i", normals[barrel], to_camera).min() > -0.02


def test_earlier_tows_stay_visible_under_the_new_one(cylinder):
    run, _, _ = cylinder
    assert set(np.unique(revolution_truth(run, 2)["revolution"])) == {-1, 0, 1, 2}


def test_true_fibre_angle_is_planned_angle_plus_error(cylinder):
    run, _, _ = cylinder
    for r in truth(run)["revolutions"]:
        k = r["revolution"]
        rev = revolution_truth(run, k)
        fibre = rev["fibre_deg"][rev["revolution"] == k]
        expected = HELIX_FIBRE_DEG + r["angle_error_deg"]
        assert np.median(fibre) == pytest.approx(expected, abs=0.05)


def test_tows_cross_the_view_at_the_specified_positions(cylinder):
    # Spec: each tow crosses the view once, at z = -120, 0 and 120 mm, facing
    # the camera on +X (body angle 0).
    run, _, _ = cylinder
    crossings = [r["crossing_point"] for r in truth(run)["revolutions"]]
    assert crossings == [pytest.approx([125, 0, z], abs=0.05) for z in (-120, 0, 120)]
