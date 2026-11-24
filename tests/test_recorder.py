"""Recording a run (slice 3) with the replay camera.

The replay camera serves the frames of the committed cylinder run; the
recorded run folder must follow the frame log rules of section 3 and be
processable by `process_run` like the committed run.
"""

import numpy as np
import pytest

from digital_thread import dataset, geometry, process
from digital_thread.camera import ReplayCamera
from digital_thread.recorder import Recorder, RecordSettings


def _settings(run, run_id="recorded"):
    return RecordSettings(
        run_id=run_id,
        calibration_folder=run.calibration_folder,
        tow_file=run.folder / run.tow_file,
        tow_width_mm=run.tow_width_mm,
        nominal_period_s=run.nominal_period_s,
        processing=run.processing,
    )


def _record(camera, folder, run, cal, check=None):
    """Capture reference, record every further frame, stop."""
    rec = Recorder(folder, _settings(run), cal, camera.polar_pattern, camera.settings())
    rec.capture_reference()
    rec.on_frame(*camera.grab())  # the next triggered frame is the reference
    rec.start()
    while (grab := camera.grab()) is not None:
        rec.on_frame(*grab)
        if check:
            check(rec)
    rec.stop()
    return rec


def test_bounce_dropped_and_missed_revolution_flagged(cylinder_geometry, tmp_path):
    run, cal, _ = cylinder_geometry
    t = {f.name: f.timestamp_s for f in run.frames}
    events = [
        ("reference", t["reference"]),
        ("rev000", t["rev000"]),
        ("rev000", t["rev000"] + 0.2),  # pin bounce
        ("rev002", t["rev002"]),  # rev001 missed
    ]
    camera = ReplayCamera(run.folder, events)
    logged = []

    def check(rec):  # run.json is readable after every frame
        logged.append([f.name for f in dataset.read_run(tmp_path).frames])

    _record(camera, tmp_path, run, cal, check)

    rec_run = dataset.read_run(tmp_path)
    assert [f.name for f in rec_run.frames] == ["reference", "rev000", "rev002"]
    assert [f.timestamp_s for f in rec_run.frames] == [0.0, 12.0, 36.0]
    assert rec_run.missed_revolutions == [1]
    assert rec_run.dropped_bounces_s == pytest.approx([12.2])
    assert logged == [
        ["reference", "rev000"],
        ["reference", "rev000"],
        ["reference", "rev000", "rev002"],
    ]
    assert rec_run.tow_file == run.tow_file
    assert (tmp_path / rec_run.tow_file).read_bytes() == (
        run.folder / run.tow_file
    ).read_bytes()


def test_pixels_outside_the_crop_polygon_are_zero(cylinder_geometry, tmp_path):
    run, cal, _ = cylinder_geometry
    camera = ReplayCamera(run.folder)
    _record(camera, tmp_path, run, cal)
    inside = np.repeat(np.repeat(geometry.crop_mask(cal), 2, 0), 2, 1)
    rec_run = dataset.read_run(tmp_path)
    for name in ("reference", "rev001"):
        raw, source = dataset.read_frame(rec_run, name), dataset.read_frame(run, name)
        assert raw.shape == (3000, 4096) and raw.dtype == np.uint16
        assert not raw[~inside].any()
        np.testing.assert_array_equal(raw[inside], source[inside])


def test_recorded_run_processes_like_the_committed_run(
    cylinder_geometry, processed_committed, tmp_path
):
    run, cal, geo = cylinder_geometry
    recorded, out = tmp_path / "recorded", tmp_path / "out"
    _record(ReplayCamera(run.folder), recorded, run, cal)
    out.mkdir()
    dataset.write_geometry(out, geo)
    rec = dataset.read_export(process.process_run(recorded, out))[0]
    committed = dataset.read_export(processed_committed("cylinder")[1])[0]
    assert len(rec) == len(committed) == 3
    for a, b in zip(committed, rec, strict=True):
        for key, value in a.items():
            if isinstance(value, float):
                np.testing.assert_allclose(b[key], value, atol=1e-9)
            else:
                assert b[key] == value
