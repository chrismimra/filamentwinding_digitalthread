"""Capture and processing stations (section 8) together in one test.

The capture station is the app with the capture role, a replay camera and a
temporary shared folder; the processing station works on the same folder, one
`poll_once()` per call.
"""

import shutil
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from digital_thread import dataset, stations
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
    """Serves the run's next frame only when the test calls `pulse()` (the pin
    passing the sensor) and the camera waits for the hardware trigger."""

    def __init__(self, run_folder):
        super().__init__(run_folder)
        self._pulses = threading.Semaphore(0)

    def pulse(self):
        self._pulses.release()

    def grab(self, timeout_s=1.0):
        if self.trigger != "hardware" or not self._pulses.acquire(timeout=0.05):
            return None  # no pin pulse
        return super().grab(timeout_s)


@pytest.fixture
def capture(cylinder_geometry, tmp_path):
    """The capture station on a shared folder, with its local spool."""
    app = QApplication.instance() or QApplication([])
    run = cylinder_geometry[0]
    share, spool = tmp_path / "share", tmp_path / "spool"
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(PinCamera(run.folder), store, share=share, spool=spool)
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
    win.camera.pulse()
    _wait(app, tab.record_button.isEnabled)
    tab.record_button.click()
    frames = len(run.frames) - 1
    for _ in range(frames):
        win.camera.pulse()
    _wait(app, lambda: tab.recorder.frame_count == frames)
    tab.stop_button.click()


def test_a_recorded_run_is_processed_and_opened(capture, processed_committed):
    app, win, run, share = capture
    station = stations.ProcessingStation(share)
    _record(app, win, run, "shared_run")
    folder = share / stations.RUNS / "shared_run"
    _wait(app, lambda: stations.job_state(folder) == "waiting")
    assert dataset.read_run(folder).calibration == "calibration"
    assert (folder / "calibration" / "calibration.json").exists()

    win.results.refresh()
    assert win.results.states()["shared_run"] == "waiting"
    assert station.poll_once() == folder
    assert stations.job_state(folder) == "done"

    win.results.refresh()
    assert win.results.states()["shared_run"] == "done"
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
