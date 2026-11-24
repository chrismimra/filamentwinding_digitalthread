"""App smoke test (slices 1-3): offscreen Qt with the replay camera."""

import time

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from digital_thread import dataset
from digital_thread.app import MainWindow
from digital_thread.camera import ReplayCamera


def _wait(app, condition, timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(cylinder_geometry, tmp_path):
    app = QApplication.instance() or QApplication([])
    run = cylinder_geometry[0]
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(ReplayCamera(run.folder, loop=True), store)
    yield app, win, run
    win.close()


def test_app_shows_the_tabs_and_a_preview_frame(window):
    app, win, _ = window
    assert win.tab_names() == ["Preview", "Calibration", "Record", "Results"]
    _wait(app, lambda: win.preview.frames > 0)
    assert not win.preview.image.pixmap().isNull()
    assert "Saturated pixels" in win.preview.status.text()


def test_record_tab_records_a_run(window, tmp_path):
    app, win, run = window
    tab = win.record
    win.preview.save_location.setText(str(tmp_path))
    tab.run_id.setText("gui_run")
    tab.calibration.setText(str(run.calibration_folder))
    tab.tow_file.setText(str(run.folder / run.tow_file))
    tab.period.setValue(run.nominal_period_s)
    win.tabs.setCurrentWidget(tab)
    tab.reference_button.click()
    _wait(app, tab.record_button.isEnabled)
    tab.record_button.click()
    _wait(app, lambda: tab.recorder.frame_count >= 2)
    tab.stop_button.click()
    recorded = dataset.read_run(tmp_path / "gui_run")
    assert recorded.frames[0].name == "reference"
    assert len(recorded.frames) >= 3


def test_results_tab_shows_the_tow_viewer(window, processed_committed):
    _, win, run = window
    out, _ = processed_committed("cylinder")
    win.results.open(run.folder, out)
    assert "Revolution 0" in win.results.viewer.text
