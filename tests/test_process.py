"""Processing a finished run into an export package (slices 4-7 end to end).

Outputs go to a temporary folder seeded with the run geometry of the session
fixtures, so processing reuses it and the committed runs stay unchanged.
Expected values come from the committed ground truth.
"""

import json

import numpy as np
import pytest

from digital_thread import dataset, process, viewer


@pytest.fixture(scope="module", params=["cylinder", "banana"])
def processed(request):
    run, cal, _ = request.getfixturevalue(f"{request.param}_geometry")
    out, export = request.getfixturevalue("processed_committed")(request.param)
    truth = json.loads((run.folder / "truth.json").read_text(encoding="utf-8"))
    return run, cal, out, export, truth["revolutions"]


def test_export_package_is_complete(processed):
    _, _, out, export, truth = processed
    assert export == out / "export"
    for name in ("tow_table.csv", "export.json"):
        assert (export / name).exists()
    for r in truth:
        k = r["revolution"]
        assert (export / dataset.overlay_file(k)).exists()
        flat = dataset.read_flat(export, k)
        assert flat.fibre_deg.shape == flat.tow.shape and flat.tow.dtype == bool
        assert flat.spacing == pytest.approx(0.1)


def test_tow_table_matches_the_truth_with_frame_timestamps(processed):
    run, _, _, export, truth = processed
    rows, _ = dataset.read_export(export)
    stamps = {f.name: f.timestamp_s for f in run.frames}
    for row, r in zip(rows, truth, strict=True):
        assert row["timestamp_s"] == stamps[f"rev{r['revolution']:03d}"]
        assert row["visible"] == r["visible"]
        if r["visible"]:
            assert row["delta_position_mm"] == pytest.approx(r["offset_mm"], abs=1.0)
            assert row["delta_angle_deg"] == pytest.approx(
                r["angle_error_deg"], abs=1.0
            )
        else:
            assert np.isnan(row["delta_position_mm"])


def test_export_json_describes_the_run(processed):
    run, cal, _, export, _ = processed
    _, meta = dataset.read_export(export)
    assert meta["run_id"] == run.run_id
    assert meta["part"] == run.calibration_folder.parent.name  # cylinder / banana
    assert meta["tow_file"] == run.tow_file
    assert meta["processing"] == {
        "dolp_threshold": 0.1,
        "incidence_limit_deg": run.processing.incidence_limit_deg,
    }
    assert meta["camera"]["centre"] == pytest.approx(-cal.R.T @ cal.t)
    assert {"frame", "units", "fibre_angle", "delta_position"} <= set(
        meta["conventions"]
    )


def test_processing_twice_gives_identical_results(cylinder_geometry, tmp_path):
    run, _, geo = cylinder_geometry
    dataset.write_geometry(tmp_path, geo)
    first = process.process_run(run.folder, tmp_path)
    rows_a, _ = dataset.read_export(first)
    flats_a = [dataset.read_flat(first, k) for k in range(len(rows_a))]
    second = process.process_run(run.folder, tmp_path)
    rows_b, _ = dataset.read_export(second)
    as_array = lambda rows: np.array([[float(v) for v in r.values()] for r in rows])
    np.testing.assert_array_equal(as_array(rows_a), as_array(rows_b))  # NaN == NaN
    for k, a in enumerate(flats_a):
        b = dataset.read_flat(second, k)
        np.testing.assert_array_equal(a.fibre_deg, b.fibre_deg)
        np.testing.assert_array_equal(a.tow, b.tow)


def test_viewer_steps_through_all_revolutions(processed):
    run, _, out, _, truth = processed
    tow_viewer = viewer.TowViewer(run.folder, out)
    rows, _ = dataset.read_export(out / "export")
    assert tow_viewer.revolution == 0
    for row, r in zip(rows, truth, strict=True):
        assert tow_viewer.revolution == r["revolution"]
        if r["visible"]:
            assert f"{row['delta_position_mm']:+.2f} mm" in tow_viewer.text
            assert f"{row['delta_angle_deg']:+.2f}" in tow_viewer.text
        else:
            assert "not visible" in tow_viewer.text
        tow_viewer.next()
    assert tow_viewer.revolution == 0  # wraps around
    tow_viewer.previous()
    assert tow_viewer.revolution == len(rows) - 1
    tow_viewer.close()
