"""Capture and processing stations (section 8) together in one test.

The capture station is the app with the capture role, a replay camera and a
temporary shared folder; the processing station works on the same folder, one
`poll_once()` per call.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from digital_thread import dataset, process, stations
from digital_thread.app import MainWindow
from digital_thread.camera import ReplayCamera
from digital_thread.recorder import Recorder, RecordSettings
from digital_thread.station import StationWindow


def _wait(app, condition, timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError
        app.processEvents()
        time.sleep(0.01)


class PinCamera(ReplayCamera):
    """No frames in free run (the preview must not use up the run); in the
    pin's software trigger mode one frame per pin pulse, so the recording
    holds exactly the committed run."""

    def grab(self, timeout_s=1.0):
        if self.trigger != "software":  # the replay camera starts in "hardware"
            time.sleep(0.01)
            return None
        return super().grab(timeout_s)


def _pulse(gpio):
    """The pin passes the normally open switch on GPIO27."""
    gpio.drive_high()
    time.sleep(0.1)  # well above the debounce, also on a busy machine
    gpio.drive_low()
    time.sleep(0.1)  # well above the debounce, also on a busy machine


@pytest.fixture
def capture(cylinder_geometry, tmp_path, gpio):
    """The capture station on a shared folder, with its local spool and the
    pin switch on (mock) GPIO27."""
    app = QApplication.instance() or QApplication([])
    run = cylinder_geometry[0]
    share, spool = tmp_path / "share", tmp_path / "spool"
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(PinCamera(run.folder), store, share=share, spool=spool)
    win.pin = gpio
    yield app, win, run, share
    win.close()


def _record(app, win, run, run_id):
    """Reference and every frame of the replayed run, then Stop."""
    tab = win.record
    tab.run_id.setText(run_id)
    tab.calibration.setText(str(run.calibration_folder))
    tab.tow_file.setText(str(run.folder / run.tow_file))
    tab.period.setValue(run.nominal_period_s)
    win.tabs.setCurrentWidget(tab)
    tab.reference_button.click()
    _pulse(win.pin)
    _wait(app, tab.record_button.isEnabled)
    tab.record_button.click()
    frames = len(run.frames) - 1
    for _ in range(frames):
        _pulse(win.pin)
    _wait(app, lambda: tab.recorder.frame_count == frames)
    tab.stop_button.click()


def _listed(win, run_id):
    """The run's state in the Results tab, as the operator sees it."""
    win.results.refresh()
    return win.results.states().get(run_id)


def test_a_recorded_run_is_processed_and_opened(capture, processed_committed):
    app, win, run, share = capture
    station = stations.ProcessingStation(share)
    _record(app, win, run, "shared_run")
    folder = share / stations.RUNS / "shared_run"
    _wait(app, lambda: stations.job_state(folder) == "waiting")
    assert dataset.read_run(folder).calibration == "calibration"
    assert (folder / "calibration" / "calibration.json").exists()

    _wait(app, lambda: _listed(win, "shared_run") == "waiting")
    assert station.poll_once() == folder
    assert stations.job_state(folder) == "done"

    _wait(app, lambda: _listed(win, "shared_run") == "done")
    assert win.tabs.currentWidget() is win.results  # opened: recorded here
    assert win.results.viewer is not None
    rows = dataset.read_export(folder / dataset.EXPORT_FOLDER)[0]
    committed = dataset.read_export(processed_committed("cylinder")[1])[0]
    assert len(rows) == len(committed)
    for a, b in zip(committed, rows, strict=True):
        for key, value in a.items():
            if isinstance(value, float):
                np.testing.assert_allclose(b[key], value, atol=1e-9)
            else:
                assert b[key] == value


def test_a_share_outage_delays_but_never_loses_frames(cylinder_geometry, tmp_path):
    run, cal, _ = cylinder_geometry
    share, spool = tmp_path / "share", tmp_path / "spool" / "outage"
    camera = ReplayCamera(run.folder)
    settings = RecordSettings(
        run_id="outage",
        calibration_folder=run.calibration_folder,
        tow_file=run.folder / run.tow_file,
        tow_width_mm=run.tow_width_mm,
        nominal_period_s=run.nominal_period_s,
    )
    rec = Recorder(spool, settings, cal, camera.polar_pattern, {})
    target = share / stations.RUNS / "outage"
    mirror = stations.Mirror(spool, target)
    rec.capture_reference()
    rec.on_frame(*camera.grab())
    rec.start()
    rec.on_frame(*camera.grab())
    with pytest.raises(OSError):
        mirror.sync()  # share not mounted (no runs folder): nothing written
    assert not share.exists()
    while (grab := camera.grab()) is not None:
        rec.on_frame(*grab)
    rec.stop()
    mirror.finish()
    (share / stations.RUNS).mkdir(parents=True)  # the share is back
    assert mirror.sync()
    assert (target / stations.JOB_FILE).exists()
    copied = dataset.read_run(target)
    assert [f.name for f in copied.frames] == [f.name for f in run.frames]
    for frame in copied.frames:
        assert (target / frame.file).read_bytes() == (spool / frame.file).read_bytes()


def test_the_spool_keeps_a_run_until_it_is_done(tmp_path):
    share, spool = tmp_path / "share", tmp_path / "spool"
    for run_id in ("done_run", "waiting_run"):
        (spool / run_id).mkdir(parents=True)
        (share / stations.RUNS / run_id).mkdir(parents=True)
        dataset.write_json(share / stations.RUNS / run_id / stations.JOB_FILE, {})
    stations.write_status(share / stations.RUNS / "done_run", "done")
    (spool / "not_copied").mkdir()
    stations.clean_spool(spool, share)
    assert sorted(p.name for p in spool.iterdir()) == ["not_copied", "waiting_run"]


def test_offline_banner_while_recording_without_the_processing_station(capture):
    app, win, run, share = capture
    _wait(app, lambda: not win.offline_banner.isHidden())
    assert "Processing station offline" in win.offline_banner.text()
    _record(app, win, run, "offline_run")  # recording does not wait for it
    assert dataset.read_run(win.spool / "offline_run").frames[-1].name == "rev002"
    station = stations.ProcessingStation(share)

    def beating():
        station.heartbeat.beat()
        return win.offline_banner.isHidden()

    _wait(app, beating)


def _shared_copy(run, share, run_id):
    """The committed run as a run job in the share, with its own bundle copy."""
    folder = share / stations.RUNS / run_id
    shutil.copytree(run.folder, folder)
    shutil.copytree(run.calibration_folder, folder / "calibration")
    copied = dataset.read_run(folder)
    copied.calibration = "calibration"
    dataset.write_run(folder, copied)
    dataset.write_json(folder / stations.JOB_FILE, {"kind": "run"})
    return folder


def test_station_window_works_through_jobs(cylinder_geometry, tmp_path):
    app = QApplication.instance() or QApplication([])
    share = tmp_path / "share"
    window = StationWindow(share)
    try:
        folder = _shared_copy(cylinder_geometry[0], share, "job_run")
        _wait(app, lambda: stations.job_state(folder) == "done", timeout_s=600)
        _wait(app, lambda: "run job_run done" in window.log.toPlainText())
        heartbeat = stations.Heartbeat(share, stations.CAPTURE_HEARTBEAT)

        def beating():
            heartbeat.beat()
            return "connected" in window.capture_state.text()

        _wait(app, beating)
    finally:
        window.close()


def test_a_failed_run_shows_its_message_on_the_capture_station(capture):
    app, win, run, share = capture
    station = stations.ProcessingStation(share)
    folder = _shared_copy(run, share, "broken_run")
    (folder / run.tow_file).unlink()
    assert station.poll_once() == folder
    assert stations.job_state(folder) == "failed"
    _wait(app, lambda: _listed(win, "broken_run") == "failed")
    text = win.results.runs.item(0).text()
    assert text.startswith("broken_run: failed") and run.tow_file in text


def test_an_interrupted_job_is_redone_at_the_next_start(cylinder_geometry, tmp_path):
    share = tmp_path / "share"
    stations.ProcessingStation(share)
    folder = _shared_copy(cylinder_geometry[0], share, "interrupted")
    stations.write_status(folder, "working", progress=[1, 4])  # then it closed
    assert stations.ProcessingStation(share).poll_once() == folder
    assert stations.job_state(folder) == "done"


IMAGES = Path(__file__).parents[1] / "demoData" / "synthetic" / "calibration_images"


def test_calibration_is_solved_on_the_processing_station(capture, tmp_path):
    _, win, _, share = capture
    truth = json.loads((IMAGES / "truth.json").read_text(encoding="utf-8"))
    station = stations.ProcessingStation(share)
    tab = win.calibration

    def solved():
        assert station.poll_once() is not None
        tab.poll_job()
        return tab.status.text()

    tab.load_images(IMAGES)
    tab.calibrate_optics()
    assert "Solving on the processing station" in tab.status.text()
    assert "Optics from 12 images" in solved()
    tab.mandrel.setText(str(IMAGES / truth["calibration"]))
    tab.load_mandrel()
    tab.marker_table.setText(str(IMAGES / truth["markers"]["table"]))
    tab.camera_position[1].setValue(250)
    tab.solve_position()
    text = solved()
    assert "Reprojection error per marker" in text and "Not on the mesh" not in text
    assert "WARNING" not in text and "Body angle facing the camera" in text
    tab.session.crop_polygon = np.array([[300, 500], [1700, 500], [1700, 1000]])
    tab.touch_down.setValue(74)
    bundle = share / stations.CALIBRATIONS / "installation"
    tab.bundle_folder.setText(str(bundle))
    tab.save()
    cal = dataset.read_calibration(bundle)
    assert cal.touch_down_deg == 74 and len(cal.crop_polygon) == 3
    local = Path(win.record.calibration.text())
    assert local != bundle and dataset.read_calibration(local).touch_down_deg == 74
    banana = dataset.read_calibration(IMAGES / truth["calibration"])
    np.testing.assert_allclose(cal.K, banana.K, rtol=0.005, atol=2.0)
    centre = -banana.R.T @ banana.t
    np.testing.assert_allclose(-cal.R.T @ cal.t, centre, atol=1.0)
    source = dataset.read_run(IMAGES.parent / "banana" / "run")
    run = tmp_path / "banana_run"  # reference and revolution 0 with the new bundle
    run.mkdir()
    shutil.copyfile(source.folder / source.tow_file, run / source.tow_file)
    source.frames = source.frames[:2]
    for frame in source.frames:
        shutil.copyfile(source.folder / frame.file, run / frame.file)
    source.calibration = str(bundle)
    dataset.write_run(run, source)
    rows, _ = dataset.read_export(process.process_run(run))
    assert [r["revolution"] for r in rows] == [0]


def test_a_failed_calibration_shows_its_message(capture):
    _, win, _, share = capture
    station = stations.ProcessingStation(share)
    tab = win.calibration
    tab.session.board_images = [np.zeros((1500, 2048), np.uint8)] * 3  # no board
    tab.calibrate_optics()
    station.poll_once()
    tab.poll_job()
    assert tab.status.text().startswith("Failed on the processing station:")


def test_runs_left_in_the_spool_are_handed_over_at_the_next_start(
    cylinder_geometry, tmp_path
):
    app = QApplication.instance() or QApplication([])
    run, cal, _ = cylinder_geometry
    share, spool = tmp_path / "share", tmp_path / "spool"
    stations.ProcessingStation(share)
    camera = ReplayCamera(run.folder)
    settings = RecordSettings("left_over", run.calibration_folder, None, 6.35, 12.0)
    rec = Recorder(spool / "left_over", settings, cal, camera.polar_pattern, {})
    rec.capture_reference()
    rec.on_frame(*camera.grab())  # then the app closed before the copy
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(ReplayCamera(run.folder), store, share=share, spool=spool)
    try:
        folder = share / stations.RUNS / "left_over"
        _wait(app, lambda: stations.job_state(folder) == "waiting")
        assert dataset.read_run(folder).frames[0].name == "reference"
    finally:
        win.close()
