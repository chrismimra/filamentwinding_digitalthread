"""Capture and processing stations (section 8) together in one test.

The capture station is the app with the capture role, a replay camera and a
temporary shared folder; the processing station works on the same folder, one
`poll_once()` per call.
"""

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from digital_thread import dataset, stations
from digital_thread.app import MainWindow
from digital_thread.camera import ReplayCamera


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
    _record(app, win, run, "shared_run")
    folder = share / stations.RUNS / "shared_run"
    _wait(app, lambda: stations.job_state(folder) == "waiting")
    assert dataset.read_run(folder).calibration == "calibration"
    assert (folder / "calibration" / "calibration.json").exists()

    win.results.refresh()
    assert win.results.states()["shared_run"] == "waiting"
    assert stations.ProcessingStation(share).poll_once() == folder
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
