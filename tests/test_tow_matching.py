"""Tow matching on both synthetic mandrels, against the committed ground truth.

truth.json gives per revolution the crossing point, whether it is visible,
the lateral offset d_k and the angle error delta_k of the as-laid tow.
"""

import json

import numpy as np
import pytest

from digital_thread import (
    conversion,
    dataset,
    geometry,
    mapping,
    overlay,
    tow_matching,
)

HELIX_FIBRE_DEG = -np.degrees(np.arctan2(2 * np.pi * 125, 360 / 3))


@pytest.fixture(scope="module", params=["cylinder", "banana"])
def mandrel(request):
    run, cal, geo = request.getfixturevalue(f"{request.param}_geometry")
    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    truth = json.loads((run.folder / "truth.json").read_text(encoding="utf-8"))
    return run, cal, geo, planned, truth["revolutions"]


def crossings(mandrel):
    run, cal, geo, planned, truth = mandrel
    return [
        tow_matching.crossing(planned, r["revolution"], cal, geo, run.processing)
        for r in truth
    ]


def test_crossing_points_match_the_truth(mandrel):
    *_, truth = mandrel
    for found, r in zip(crossings(mandrel), truth, strict=True):
        assert np.linalg.norm(found.point - r["crossing_point"]) < 1.0
        assert found.visible == r["visible"]


def test_planned_angle_on_the_cylinder_is_the_helix_angle(cylinder_geometry):
    run, cal, geo = cylinder_geometry
    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    for k in range(3):
        found = tow_matching.crossing(planned, k, cal, geo, run.processing)
        assert found.planned_deg == pytest.approx(HELIX_FIBRE_DEG, abs=0.05)


def fibre_map(run, cal, geo, name):
    maps = conversion.convert(dataset.read_frame(run, name), run.polar_pattern)
    return mapping.fibre_angles(maps, geo, cal, run.processing)


def test_measured_tow_is_the_new_tow_only(cylinder_geometry):
    run, cal, geo = cylinder_geometry
    before, after = (fibre_map(run, cal, geo, n) for n in ("rev001", "rev002"))
    measured = tow_matching.measured_tow(before, after, geo)
    revolution = np.load(run.folder / "rev002_truth.npz")["revolution"]
    flat_revolution = mapping.flatten(revolution.astype(float), geo)
    flat_valid = np.isfinite(mapping.flatten(after, geo))
    new = (flat_revolution == 2) & flat_valid
    assert measured[new].mean() > 0.95
    assert not measured[(flat_revolution == 0) | (flat_revolution == 1)].any()


@pytest.fixture(scope="module")
def table(mandrel):
    run, cal, geo, planned, truth = mandrel
    fibre = {f.name: fibre_map(run, cal, geo, f.name) for f in run.frames}
    return tow_matching.tow_table(run, cal, geo, planned, fibre), truth


def test_tow_table_reports_the_true_deviations(table):
    rows, truth = table
    assert [row["revolution"] for row in rows] == [r["revolution"] for r in truth]
    for row, r in zip(rows, truth, strict=True):
        assert row["visible"] == r["visible"]
        if r["visible"]:
            assert row["delta_position_mm"] == pytest.approx(r["offset_mm"], abs=1.0)
            assert row["delta_angle_deg"] == pytest.approx(
                r["angle_error_deg"], abs=1.0
            )
            assert row["tow_cells"] > 1000
            if abs(r["angle_error_deg"]) >= 0.3:  # guards the sign of Delta angle
                assert row["delta_angle_deg"] == pytest.approx(
                    r["angle_error_deg"], abs=0.3
                )
        else:
            assert np.isnan(row["delta_position_mm"]) and np.isnan(
                row["delta_angle_deg"]
            )


def test_tow_table_round_trip(tmp_path, table):
    rows, _ = table
    dataset.write_tow_table(tmp_path, rows)
    back = dataset.read_tow_table(tmp_path)
    assert [r["visible"] for r in back] == [r["visible"] for r in rows]
    for got, want in zip(back, rows, strict=True):
        for key, value in want.items():
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(got[key])
            else:
                assert got[key] == pytest.approx(value)


def test_overlay_shows_the_measured_tow_and_the_planned_edges(cylinder_geometry):
    run, cal, geo = cylinder_geometry
    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    before, after = (fibre_map(run, cal, geo, n) for n in ("rev001", "rev002"))
    tow = tow_matching.measured_tow(before, after, geo)
    image = overlay.overlay_image(after, tow, planned, 2, cal, geo, run.tow_width_mm)
    assert image.shape == (*geo.flat_pixel.shape, 3) and image.dtype == np.uint8
    tow_coloured = np.all(image == overlay.TOW_COLOUR, axis=-1)
    edges = np.all(image == overlay.PLANNED_COLOUR, axis=-1)  # drawn last
    assert edges.sum() > 1000
    assert (tow_coloured | edges)[tow].all()
    assert not tow_coloured[~tow].any()


def test_a_revolution_that_never_reaches_the_camera_plane_has_no_crossing(
    cylinder_geometry,
):
    run, cal, geo = cylinder_geometry
    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    # Keep only revolution 1's path away from its crossing (z = 0): > 40 mm off.
    far = (planned.revolution != 1) | (np.abs(planned.point[:, 2]) > 40)
    partial = geometry.PlannedPath(
        planned.point[far], planned.normal[far], planned.revolution[far]
    )
    found = tow_matching.crossing(partial, 1, cal, geo, run.processing)
    assert not found.visible and np.isnan(found.point).all()
