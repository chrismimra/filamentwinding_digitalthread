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
    store.setValue("pin_input", "line")  # this computer has no GPIO
    win = MainWindow(ReplayCamera(run.folder, loop=True), store)
    yield app, win, run
    win.close()


def _press(gpio):
    """The pin passes the normally open switch on GPIO27 (with contact chatter)."""
    for _ in range(3):
        gpio.drive_high()
        gpio.drive_low()
    gpio.drive_high()
    time.sleep(0.1)  # well above the debounce, also on a busy machine
    gpio.drive_low()
    time.sleep(0.1)  # well above the debounce, also on a busy machine


def _start_reference(app, win, run, tmp_path, run_id):
    tab = win.record
    win.preview.save_location.setText(str(tmp_path))
    tab.run_id.setText(run_id)
    tab.calibration.setText(str(run.calibration_folder))
    tab.tow_file.setText(str(run.folder / run.tow_file))
    tab.period.setValue(run.nominal_period_s)
    win.tabs.setCurrentWidget(tab)
    tab.reference_button.click()
    return tab


def test_record_tab_records_through_the_gpio_pin(window, gpio, tmp_path):
    app, win, run = window
    win.preview.pin_input.setCurrentIndex(win.preview.pin_input.findData("gpio"))
    tab = _start_reference(app, win, run, tmp_path, "gpio_run")
    _press(gpio)
    _wait(app, tab.record_button.isEnabled)
    tab.record_button.click()
    for _ in range(2):
        _press(gpio)
    _wait(app, lambda: tab.recorder.frame_count == 2)
    time.sleep(0.3)
    app.processEvents()
    assert tab.recorder.frame_count == 2  # chatter gave no extra frames
    tab.stop_button.click()
    recorded = dataset.read_run(tmp_path / "gpio_run")
    assert recorded.camera_settings["pin_input"] == "gpio"
    assert recorded.frames[0].name == "reference"


def test_without_gpio_recording_says_so(cylinder_geometry, no_gpio, tmp_path):
    app = QApplication.instance() or QApplication([])
    run = cylinder_geometry[0]
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(ReplayCamera(run.folder, loop=True), store)  # default: GPIO27
    try:
        tab = _start_reference(app, win, run, tmp_path, "no_gpio_run")
        assert "No GPIO27" in tab.status.text()
        assert not tab.record_button.isEnabled() and win.grabber.mode == "free"
    finally:
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


def test_calibration_tab_shows_a_live_view(window):
    app, win, _ = window
    tab = win.calibration
    win.tabs.setCurrentWidget(tab)
    _wait(app, lambda: tab.live_frames > 0)
    assert tab.board_hint.text() == "Board not found"  # cylinder frames: no board
    tab.live.setChecked(False)
    frames = tab.live_frames
    for _ in range(50):  # let further frames arrive
        app.processEvents()
        time.sleep(0.01)
    assert tab.live_frames <= frames + 1  # at most one frame already on its way


def test_without_gpio_the_pin_input_stays_on_the_camera_line(window, no_gpio):
    _, win, _ = window
    choice = win.preview.pin_input
    choice.setCurrentIndex(choice.findData("gpio"))
    assert "No GPIO27" in win.preview.status.text()
    assert choice.currentData() == "line" and win.grabber.pin_input == "line"
    assert win.store.value("pin_input") == "line"


def test_the_pin_input_cannot_change_during_a_recording(window, tmp_path):
    app, win, run = window
    tab = _start_reference(app, win, run, tmp_path, "locked_run")
    choice = win.preview.pin_input
    choice.setCurrentIndex(choice.findData("gpio"))
    assert "stop the recording" in win.preview.status.text()
    assert choice.currentData() == "line" and win.grabber.mode == "hardware"
    tab.stop_button.setEnabled(True)
    tab.stop_button.click()


class LosingCamera(ReplayCamera):
    """A replay camera that can drop off the USB bus (`gone = True`)."""

    gone = False

    def grab(self, timeout_s=1.0):
        if self.gone:
            raise RuntimeError("Device has been removed from the PC.")
        return super().grab(timeout_s)


@pytest.fixture
def losing_window(cylinder_geometry, tmp_path):
    app = QApplication.instance() or QApplication([])
    run = cylinder_geometry[0]
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store.setValue("pin_input", "line")  # this computer has no GPIO
    camera = LosingCamera(run.folder, loop=True)
    win = MainWindow(camera, store)
    yield app, win, run, camera
    win.close()


def test_a_lost_camera_shows_a_banner(losing_window):
    app, win, _, camera = losing_window
    camera.gone = True
    _wait(app, lambda: not win.camera_banner.isHidden())
    assert "Camera lost" in win.camera_banner.text()
    assert "Device has been removed" in win.camera_banner.text()


def test_a_lost_camera_ends_the_recording(losing_window, tmp_path):
    app, win, run, camera = losing_window
    tab = _start_reference(app, win, run, tmp_path, "lost_run")
    _wait(app, lambda: tab.recorder.state == "ready")
    tab.record_button.click()
    camera.gone = True
    _wait(app, lambda: not tab.active())
    assert "Camera lost: the recording broke off" in tab.status.text()
    assert tab.recorder.folder.exists()
