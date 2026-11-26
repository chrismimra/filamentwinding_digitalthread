"""Monitoring app (production, slices 1-3): Preview, Calibration, Record, Results.

Command: `uv run python -m digital_thread.app [--replay <run folder>]
[--role capture --share <folder> [--spool <folder>]]`
With `--replay`, a replay camera serves the frames of a (synthetic) run instead
of the Basler camera. With the capture role (section 8), runs are recorded into
a local spool, copied to the shared folder and processed by the processing
station; the Results tab lists the runs in the shared folder. The tabs are thin; the logic sits in the camera,
preview, calibration, recorder, process and viewer modules.
"""

import argparse
import functools
import inspect
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from digital_thread import (
    calibration,
    conversion,
    dataset,
    preview,
    process,
    stations,
)
from digital_thread.camera import BaslerCamera, ReplayCamera
from digital_thread.pin import GpioPin
from digital_thread.recorder import Recorder, RecordSettings
from digital_thread.viewer import TowViewer

TOUCH_FONT = 14
MIRROR_INTERVAL_S = 0.5  # how often new files are copied to the shared folder
RESULTS_REFRESH_MS = 2000  # how often the Results tab reads the run states
JOB_POLL_MS = 500  # how often the Calibration tab checks a pending job
BANNER_REFRESH_MS = 500  # how often the offline banner is updated
PIN_INPUTS = {"gpio": "Pi GPIO2", "line": "Camera input line"}  # section 3
PREVIEW_WIDTH = 1024  # displayed width of camera images (super-pixel image halved)


def touch_button(text):
    button = QPushButton(text)
    button.setMinimumHeight(64)
    return button


def spin(value, low, high, decimals=2, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setMinimumHeight(48)
    return box


def path_field(dialog, directory=True, file_filter=""):
    """Line edit with a Browse button; returns (widget, line edit)."""
    edit = QLineEdit()
    edit.setMinimumHeight(48)
    button = QPushButton("Browse")
    button.setMinimumHeight(48)

    def browse():
        if directory:
            chosen = QFileDialog.getExistingDirectory(None, dialog, edit.text())
        else:
            chosen, _ = QFileDialog.getOpenFileName(
                None, dialog, edit.text(), file_filter
            )
        if chosen:
            edit.setText(chosen)

    button.clicked.connect(browse)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit)
    layout.addWidget(button)
    return row, edit


def to_pixmap(gray, colour=None):
    """8-bit grey image (+ optional RGB overlay mask -> colour) as a scaled pixmap."""
    rgb = np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)
    for mask, value in colour or []:
        rgb[mask] = value
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaledToWidth(
        PREVIEW_WIDTH, Qt.TransformationMode.SmoothTransformation
    )


def reported(method):
    """Run a tab step; an error is shown in the tab's status line, not raised."""

    takes = len(inspect.signature(method).parameters) - 1  # without self

    @functools.wraps(method)
    def step(self, *args):  # Qt may append a `checked` argument
        try:
            return method(self, *args[:takes])
        except (ValueError, OSError, RuntimeError) as error:
            self.status.setText(f"Error: {error}")

    return step


class Grabber(QThread):
    """Grabs frames in a worker thread and hands them to the GUI as signals.

    Each frame carries the trigger generation it was grabbed in; frames from
    before a trigger switch are dropped by the GUI (`current`). In free run a
    new frame is only emitted once the GUI has handled the previous one
    (`handled`), so a slow GUI never builds up a queue; triggered frames are
    never skipped.
    """

    frame = Signal(object, float, int)

    def __init__(self, camera, pin_input="gpio"):
        super().__init__()
        self.camera = camera
        self.lock = threading.Lock()  # camera calls from GUI and worker thread
        self.mode = "free"
        self.pin_input = pin_input  # a key of PIN_INPUTS
        self.generation = 0
        self._pending = threading.Event()  # a free-run frame awaits the GUI
        self._running = True
        self._gpio = None  # the GPIO pin while triggered through it
        self.trigger_errors = 0  # pin pulses the camera could not take

    def run(self):
        while self._running:
            with self.lock:
                grab = self.camera.grab(timeout_s=0.2)
                generation, mode = self.generation, self.mode
            if grab is None:
                self.msleep(20)
            elif mode != "free" or not self._pending.is_set():
                if mode == "free":
                    self._pending.set()
                self.frame.emit(*grab, generation)

    def handled(self):
        self._pending.clear()

    def current(self, generation):
        return generation == self.generation

    def set_trigger(self, mode):
        """ "free" (Preview) or "hardware" (one frame per pin pulse). The pin
        reaches the camera through its input line, or through GPIO2 of the
        Pi and a software trigger (`pin_input`)."""
        with self.lock:
            if mode == self.mode:
                return  # keep frames already on their way
            gpio = mode == "hardware" and self.pin_input == "gpio"
            self._close_gpio()  # no pulse reaches a camera being reconfigured
            self.camera.set_trigger("software" if gpio else mode)
            if gpio:
                try:
                    self._gpio = GpioPin(self._fire)
                except RuntimeError:  # no GPIO: back to the previous mode
                    self.camera.set_trigger(self.mode)
                    raise
                self.trigger_errors = 0
            self.mode = mode
            self.generation += 1
            self._pending.clear()

    def set_pin_input(self, pin_input):
        """Only between recordings; GPIO2 is checked right away."""
        if self.mode == "hardware":
            raise RuntimeError("stop the recording or marker capture first")
        if pin_input == "gpio":
            GpioPin(lambda: None).close()  # raises without GPIO
        self.pin_input = pin_input

    def _fire(self):
        """A GPIO pin pulse (gpiozero's thread). Not under the lock: the grab
        loop holds it while waiting for this very frame."""
        try:
            self.camera.software_trigger()
        except Exception:  # noqa: BLE001 (thread boundary: counted, shown at Stop)
            self.trigger_errors += 1

    def _close_gpio(self):
        if self._gpio is not None:
            self._gpio.close()
            self._gpio = None

    def stop(self):
        self._running = False
        self.wait(10000)
        self._close_gpio()


class PreviewTab(QWidget):
    """Live S0 with saturation warning (red) and focus aid (green edges)."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        settings = window.camera.settings()
        self.image = QLabel("No frame yet")
        self.image.setMinimumSize(PREVIEW_WIDTH, 750)
        self.status = QLabel()
        self.exposure = spin(settings.get("exposure_us", 2000), 20, 1e6, 0, " µs")
        self.gain = spin(settings.get("gain_db", 0.0), 0, 24, 1, " dB")
        self.show_edges = QCheckBox("Focus aid")
        self.show_edges.setMinimumHeight(48)
        self.exposure.valueChanged.connect(window.set_exposure)
        self.gain.valueChanged.connect(window.set_gain)
        save_row, self.save_location = path_field("Save location of run folders")
        self.save_location.setText(
            window.store.value("save_location", str(Path.home()))
        )
        if window.share is not None:  # capture role: runs go to the local spool
            self.save_location.setText(str(window.spool))
            save_row.setEnabled(False)
        self.save_location.textChanged.connect(
            lambda text: window.store.setValue("save_location", text)
        )
        self.pin_input = QComboBox()
        self.pin_input.setMinimumHeight(48)
        for key, text in PIN_INPUTS.items():
            self.pin_input.addItem(text, key)
        self.pin_input.setCurrentIndex(
            self.pin_input.findData(window.store.value("pin_input", "gpio"))
        )
        self.pin_input.currentIndexChanged.connect(self.choose_pin_input)
        form = QFormLayout()
        form.addRow("Exposure", self.exposure)
        form.addRow("Gain", self.gain)
        form.addRow("Pin input", self.pin_input)
        form.addRow("Save location", save_row)
        form.addRow(self.show_edges)
        form.addRow(self.status)
        layout = QHBoxLayout(self)
        layout.addWidget(self.image, stretch=3)
        side = QWidget()
        side.setLayout(form)
        layout.addWidget(side, stretch=1)
        self.frames = 0

    def choose_pin_input(self, _index=None):
        key = self.pin_input.currentData()
        grabber = self.window.grabber
        try:
            grabber.set_pin_input(key)
        except RuntimeError as error:
            self.pin_input.blockSignals(True)  # back to the pin input in use
            self.pin_input.setCurrentIndex(self.pin_input.findData(grabber.pin_input))
            self.pin_input.blockSignals(False)
            self.status.setText(f"Error: {error}")
            return
        self.window.store.setValue("pin_input", key)
        self.status.setText(f"Pin input: {PIN_INPUTS[key]}")

    def show_frame(self, raw):
        s0 = conversion.convert(raw, self.window.camera.polar_pattern).s0
        mask, count = preview.saturation(raw)
        value, edges = preview.focus(s0)
        overlays = [(mask, (255, 0, 0))]
        if self.show_edges.isChecked():
            overlays.insert(0, (edges, (0, 255, 0)))
        self.image.setPixmap(to_pixmap(calibration.to_uint8(s0), overlays))
        warning = "  SATURATED" if count else ""
        self.status.setText(f"Saturated pixels: {count}{warning}\nFocus: {value:.0f}")
        self.frames += 1


# Guidance for the 12 board positions of the intrinsic calibration.
BOARD_POSITIONS = [
    "board in the centre, facing the camera",
    "top left corner of the image",
    "top right corner",
    "bottom left corner",
    "bottom right corner",
    "centre, tilted left (about 30°)",
    "centre, tilted right",
    "centre, tilted up",
    "centre, tilted down",
    "close to the camera, filling the image",
    "far from the camera, rotated about 20°",
    "left edge, tilted towards the centre",
]


def vector_fields(values):
    """Three spin boxes (x, y, z) in one row."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    boxes = [spin(v, -1e5, 1e5, 1) for v in values]
    for box in boxes:
        layout.addWidget(box)
    return row, boxes


class CalibrationTab(QWidget):
    """Intrinsic optics (12 board images) and position (markers at the pin pulse).

    Images come from the camera (Capture buttons) or from a folder of 8-bit S0
    PNGs (board_*.png, markers_*.png). The crop polygon is drawn by clicking on
    the marker image; the bundle is written with pin phase 0.
    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.session = calibration.Session()
        self._capture = None  # "board" (next frame) or "markers" (next 5 pulses)
        self._shown = None  # kind of the image on the canvas
        self._job = None  # capture role: (kind, folder) of the pending job
        self.job_timer = QTimer(self)
        self.job_timer.timeout.connect(self.poll_job)

        self.board_button = touch_button("Capture board image")
        self.board_button.clicked.connect(lambda: self._start_capture("board"))
        self.board_button.setToolTip(f"First position: {BOARD_POSITIONS[0]}")
        self.optics_button = touch_button("Calibrate optics")
        self.optics_button.clicked.connect(self.calibrate_optics)
        images_row, self.image_folder = path_field("Folder with calibration images")
        load_images = touch_button("Load images from folder")
        load_images.clicked.connect(lambda: self.load_images(self.image_folder.text()))

        mandrel_row, self.mandrel = path_field("STEP file or calibration bundle", False)
        self.load_mandrel_button = touch_button("Load mandrel")
        self.load_mandrel_button.clicked.connect(self.load_mandrel)
        axis_point_row, self.axis_point = vector_fields((0, 0, 0))
        axis_dir_row, self.axis_dir = vector_fields((0, 1, 0))
        table_row, self.marker_table = path_field("Marker table", False, "CSV (*.csv)")
        self.marker_size = spin(30.0, 1, 500, 1, " mm")
        camera_row, self.camera_position = vector_fields((0, 0, -700))
        self.markers_button = touch_button("Capture markers at the pin (5 revolutions)")
        self.markers_button.clicked.connect(lambda: self._start_capture("markers"))
        self.position_button = touch_button("Solve position")
        self.position_button.clicked.connect(self.solve_position)
        self.touch_down = spin(0.0, -360, 360, 1, " °")
        clear_polygon = QPushButton("Clear crop polygon")
        clear_polygon.setMinimumHeight(48)
        clear_polygon.clicked.connect(self.clear_polygon)
        bundle_row, self.bundle_folder = path_field("Calibration bundle folder")
        if window.share is not None:
            self.bundle_folder.setText(
                str(
                    window.share
                    / stations.CALIBRATIONS
                    / f"bundle_{datetime.now():%Y%m%d}"
                )
            )
        self.save_button = touch_button("Save calibration bundle")
        self.save_button.clicked.connect(self.save)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.live = QCheckBox("Live view")
        self.live.setMinimumHeight(48)
        self.live.setChecked(True)
        self.board_hint = QLabel()
        self.live_frames = 0

        form = QFormLayout()
        form.addRow(QLabel("<b>A) Intrinsic optics</b>"))
        form.addRow(self.live, self.board_hint)
        form.addRow(self.board_button)
        form.addRow("Image folder", images_row)
        form.addRow(load_images)
        form.addRow(self.optics_button)
        form.addRow(QLabel("<b>B) Position</b>"))
        form.addRow("Mandrel", mandrel_row)
        form.addRow("Axis point", axis_point_row)
        form.addRow("Axis direction", axis_dir_row)
        form.addRow(self.load_mandrel_button)
        form.addRow("Marker table", table_row)
        form.addRow("Marker size", self.marker_size)
        form.addRow("Camera position", camera_row)
        form.addRow(self.markers_button)
        form.addRow(self.position_button)
        form.addRow("Touch-down angle", self.touch_down)
        form.addRow(clear_polygon)
        form.addRow("Bundle folder", bundle_row)
        form.addRow(self.save_button)
        form.addRow(self.status)
        side = QWidget()
        side.setLayout(form)

        self.figure = Figure(figsize=(10, 7.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.axis("off")
        self.canvas.mpl_connect("button_press_event", self._click)
        layout = QHBoxLayout(self)
        layout.addWidget(side, stretch=1)
        layout.addWidget(self.canvas, stretch=2)

    # Images
    @reported
    def _start_capture(self, kind):
        if kind == "markers":
            self.window.set_trigger("hardware")  # fails here without the pin input
            self.session.marker_images = []
            self.status.setText("Waiting for 5 pin pulses ...")
        self._capture = kind

    def on_frame(self, raw, timestamp_s):
        showing = self.window.tabs.currentWidget() is self and self.live.isChecked()
        if self._capture is None and not showing:
            return
        s0 = conversion.convert(raw, self.window.camera.polar_pattern).s0
        image = calibration.to_uint8(s0)
        if self._capture is None:
            self.show_live(image)
            return
        if self._capture == "board":
            self.session.board_images.append(image)
            self._capture = None
            n = len(self.session.board_images)
            text = f"Board images: {n}/{calibration.INTRINSIC_IMAGES}"
            if n < len(BOARD_POSITIONS):
                text += f"\nNext: {BOARD_POSITIONS[n]}"
            self.status.setText(text)
            if not self.live.isChecked():
                self.show_image(image, "board")
            return
        self.session.marker_images.append(image)
        n = len(self.session.marker_images)
        self.status.setText(f"Marker images: {n}/{calibration.POSITION_IMAGES}")
        if n == calibration.POSITION_IMAGES:
            self._capture = None
            self.window.release_trigger()
        self.live.setChecked(False)  # keep the marker image for the crop polygon
        self.show_image(image, "markers")

    def show_live(self, image):
        """Live camera image (half resolution) with the board corners if found."""
        corners = calibration.find_board(image)
        h, w = image.shape
        if self._shown != "live":
            self._shown = "live"
            self.ax.clear()
            self.ax.axis("off")
            self._live_image = self.ax.imshow(
                image[::2, ::2],
                cmap="gray",
                vmin=0,
                vmax=255,
                extent=(-0.5, w - 0.5, h - 0.5, -0.5),
            )
            (self._live_corners,) = self.ax.plot([], [], "o", color="lime", ms=4)
        else:
            self._live_image.set_data(image[::2, ::2])
        xy = corners if corners is not None else np.empty((0, 2))
        self._live_corners.set_data(xy[:, 0], xy[:, 1])
        n = len(self.session.board_images)
        self.board_hint.setText(
            f"Board found ({n}/{calibration.INTRINSIC_IMAGES} captured)"
            if corners is not None
            else "Board not found"
        )
        self.live_frames += 1
        self.canvas.draw_idle()

    def capturing(self):
        """True while marker images are taken at the pin pulse."""
        return self._capture == "markers"

    @reported
    def load_images(self, folder):
        folder = Path(folder)
        self.session.board_images = [
            dataset.read_image(f) for f in sorted(folder.glob("board_*.png"))
        ]
        self.session.marker_images = [
            dataset.read_image(f) for f in sorted(folder.glob("markers_*.png"))
        ]
        self.status.setText(
            f"Loaded {len(self.session.board_images)} board and "
            f"{len(self.session.marker_images)} marker images"
        )
        if self.session.marker_images:
            self.live.setChecked(False)
            self.show_image(self.session.marker_images[0], "markers")

    def show_image(self, image, kind):
        """Show a board or marker image; the crop polygon is drawn on the latter."""
        self._shown = kind
        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        self._draw_polygon()

    # Steps
    @reported
    def calibrate_optics(self):
        if self.window.share is not None:
            if not self.session.board_images:
                raise ValueError("capture or load the board images first")
            self._submit(
                "optics",
                stations.submit_optics(self.window.share, self.session.board_images),
            )
            return
        self._show_optics(self.session.calibrate_optics())

    def _show_optics(self, result):
        self.status.setText(
            f"Optics from {result.used} images: f = {result.K[0, 0]:.1f} px, "
            f"principal point ({result.K[0, 2]:.1f}, {result.K[1, 2]:.1f}), "
            f"k1 = {result.distortion[0]:.4f}, reprojection error {result.rms_px:.2f} px"
        )

    def _submit(self, kind, folder):
        """Capture role: a calibration job, solved on the processing station."""
        self._job = (kind, folder)
        self.status.setText("Solving on the processing station ...")
        self.job_timer.start(JOB_POLL_MS)

    @reported
    def poll_job(self):
        """Capture role: show the result once the processing station is done."""
        if self._job is None:
            return
        kind, folder = self._job
        state = stations.job_state(folder)
        if state not in stations.FINISHED:
            return
        self._job = None
        self.job_timer.stop()
        if state == "failed":
            message = stations.read_json(folder / stations.STATUS_FILE)["message"]
            self.status.setText(f"Failed on the processing station: {message}")
        elif kind == "optics":
            self.session.intrinsics = stations.read_optics(folder)
            self._show_optics(self.session.intrinsics)
        else:
            stations.read_position(folder, self.session)
            self._show_position()

    @reported
    def load_mandrel(self):
        source = Path(self.mandrel.text())
        if source.is_dir():  # an existing bundle: its mesh and axis
            self.session.load_mandrel(source)
            for box, v in zip(self.axis_point, self.session.axis_point, strict=True):
                box.setValue(v)
            for box, v in zip(self.axis_dir, self.session.axis_dir, strict=True):
                box.setValue(v)
        elif self.window.share is not None:  # gmsh runs on the processing station
            if not source.is_file():
                raise ValueError(f"{source} is not a file")
            self.status.setText("The mesh is made on the processing station.")
            return
        else:
            point = [box.value() for box in self.axis_point]
            direction = [box.value() for box in self.axis_dir]
            self.session.load_mandrel(source, point, direction)
        self.status.setText(f"Mesh: {len(self.session.mesh[1])} triangles")

    @reported
    def solve_position(self):
        session = self.session
        session.marker_table = Path(self.marker_table.text())
        session.marker_size_mm = self.marker_size.value()
        if self.window.share is not None:
            self._submit(
                "position",
                stations.submit_position(
                    self.window.share,
                    session,
                    self.mandrel.text(),
                    [box.value() for box in self.axis_point],
                    [box.value() for box in self.axis_dir],
                ),
            )
            return
        session.solve_position()
        self._show_position()

    def _show_position(self):
        session, pose = self.session, self.session.pose
        lines = []
        off = session.markers_off_mesh()
        if off:
            lines.append(f"Not on the mesh (ignored): markers {off}")
        errors = ", ".join(f"{i}: {e:.2f}" for i, e in pose.marker_errors_px.items())
        lines.append(f"Reprojection error per marker [px]: {errors}")
        centre = ", ".join(f"{c:.1f}" for c in pose.centre)
        lines.append(f"Camera position: ({centre}) mm")
        entered = [box.value() for box in self.camera_position]
        mismatch = calibration.position_mismatch(pose, entered)
        if mismatch is not None:
            lines.append(f"WARNING: {mismatch:.0f} mm from the entered position")
        guide = session.body_angle_guide()
        lines.append(f"Body angle facing the camera: {guide:.1f}° (touch-down guide)")
        self.status.setText("\n".join(lines))

    # Crop polygon
    def _click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self._shown != "markers":  # the polygon belongs on the empty mandrel
            self.status.setText("Draw the crop polygon on a marker image.")
            return
        vertex = np.array([[event.xdata, event.ydata]])
        polygon = self.session.crop_polygon
        self.session.crop_polygon = (
            vertex if polygon is None else np.vstack([polygon, vertex])
        )
        self._draw_polygon()

    def clear_polygon(self):
        self.session.crop_polygon = None
        if self.session.marker_images:
            self.live.setChecked(False)
            self.show_image(self.session.marker_images[0], "markers")

    def _draw_polygon(self):
        polygon = self.session.crop_polygon
        if polygon is not None:
            closed = np.vstack([polygon, polygon[:1]])
            self.ax.plot(closed[:, 0], closed[:, 1], "-o", color="tab:orange")
        self.canvas.draw_idle()

    @reported
    def save(self):
        self.session.touch_down_deg = self.touch_down.value()
        folder = Path(self.bundle_folder.text())
        self.session.save(folder)
        text = f"Calibration bundle written to {folder}"
        if self.window.share is not None:  # a local copy, so recording needs no share
            local = self.window.spool.parent / stations.CALIBRATIONS / folder.name
            self.session.save(local)
            self.window.record.calibration.setText(str(local))
            text += f"\nLocal copy for recording: {local}"
        self.status.setText(text)


class RecordTab(QWidget):
    """Capture reference, Record, Stop; frames go to the recorder."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.recorder = None
        store = window.store
        self.run_id = QLineEdit()
        self.run_id.setMinimumHeight(48)
        cal_row, self.calibration = path_field("Calibration bundle")
        self.calibration.setText(store.value("calibration", ""))
        tow_row, self.tow_file = path_field("Tow file", False, "CSV (*.csv)")
        self.tow_file.setText(store.value("tow_file", ""))
        self.tow_width = spin(6.35, 0.1, 100, 2, " mm")
        self.period = spin(float(store.value("period", 12.0)), 0.1, 3600, 2, " s")
        self.dolp = spin(0.1, 0, 1, 2)
        self.incidence = spin(20.0, 0, 89, 1, " °")
        self.auto_process = QCheckBox("Process the run after Stop")
        self.auto_process.setMinimumHeight(48)
        self.auto_process.setVisible(window.share is None)  # capture role: always
        self.reference_button = touch_button("Capture reference")
        self.record_button = touch_button("Record")
        self.stop_button = touch_button("Stop")
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.reference_button.clicked.connect(self.capture_reference)
        self.record_button.clicked.connect(self.record)
        self.stop_button.clicked.connect(self.stop)
        self.status = QLabel("Idle")
        self.last = QLabel()
        form = QFormLayout()
        form.addRow("Run ID", self.run_id)
        form.addRow("Calibration bundle", cal_row)
        form.addRow("Tow file", tow_row)
        form.addRow("Tow width", self.tow_width)
        form.addRow("Nominal period", self.period)
        form.addRow("DoLP threshold", self.dolp)
        form.addRow("Incidence limit", self.incidence)
        form.addRow(self.auto_process)
        buttons = QHBoxLayout()
        for button in (self.reference_button, self.record_button, self.stop_button):
            buttons.addWidget(button)
        left = QVBoxLayout()
        left.addLayout(form)
        left.addLayout(buttons)
        left.addWidget(self.status)
        left.addStretch()
        layout = QHBoxLayout(self)
        layout.addLayout(left, stretch=1)
        layout.addWidget(self.last, stretch=2)

    def active(self):
        """True from Capture reference until Stop (the pin trigger is needed)."""
        return self.recorder is not None and self.recorder.state in (
            "reference",
            "ready",
            "recording",
        )

    def run_folder(self):
        run_id = self.run_id.text() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id.setText(run_id)
        return Path(self.window.preview.save_location.text()) / run_id

    @reported
    def capture_reference(self):
        folder = self.run_folder()
        shared = self.window.shared_run(folder.name)
        if folder.exists() or (shared is not None and shared.exists()):
            QMessageBox.warning(
                self, "Run exists", f"{folder} exists; change the run ID."
            )
            return
        cal_folder = Path(self.calibration.text())
        cal = dataset.read_calibration(cal_folder)
        store = self.window.store
        store.setValue("calibration", str(cal_folder))
        store.setValue("tow_file", self.tow_file.text())
        store.setValue("period", self.period.value())
        settings = RecordSettings(
            run_id=self.run_id.text(),
            calibration_folder=cal_folder,
            tow_file=Path(self.tow_file.text()) if self.tow_file.text() else None,
            tow_width_mm=self.tow_width.value(),
            nominal_period_s=self.period.value(),
            processing=dataset.Processing(self.dolp.value(), self.incidence.value()),
        )
        camera = self.window.camera
        self.recorder = Recorder(
            folder,
            settings,
            cal,
            camera.polar_pattern,
            {**camera.settings(), "pin_input": self.window.grabber.pin_input},
        )
        self.window.set_trigger("hardware")
        self.recorder.capture_reference()
        self.window.mirror(folder)
        self.status.setText("Waiting for the next pin pulse (reference) ...")

    @reported
    def record(self):
        self.recorder.start()
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Recording: 0 frames")

    def stop(self):
        self.recorder.stop()
        self.stop_button.setEnabled(False)
        self.window.release_trigger()
        run = self.recorder.run
        text = f"Stopped: {self.recorder.frame_count} frames in {self.recorder.folder}"
        if run.missed_revolutions:
            text += f"\nMissed revolutions: {run.missed_revolutions}"
        failed = self.window.grabber.trigger_errors
        if failed:
            text += f"\nPin pulses the camera could not take: {failed}"
        if self.window.share is not None:
            self.window.hand_over(self.recorder.folder)
            text += "\nCopying to the processing station ..."
        elif self.auto_process.isChecked():
            self.window.process(self.recorder.folder)
        self.status.setText(text)

    def on_frame(self, raw, timestamp_s):
        if self.recorder is None:
            return
        name = self.recorder.on_frame(raw, timestamp_s)
        if name is None:
            return
        if name == "reference":
            self.record_button.setEnabled(True)
            self.status.setText("Reference captured. Press Record.")
        else:
            run = self.recorder.run
            missed = (
                f", missed {run.missed_revolutions}" if run.missed_revolutions else ""
            )
            self.status.setText(
                f"Recording: {self.recorder.frame_count} frames{missed}"
            )
        s0 = conversion.convert(raw, self.window.camera.polar_pattern).s0
        self.last.setPixmap(to_pixmap(calibration.to_uint8(s0)))


class ResultsTab(QWidget):
    """The tow viewer of a processed run, embedded in a Qt canvas. With the
    capture role, also the list of runs in the shared folder with their state."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        run_row, self.run_folder = path_field("Processed run folder")
        open_button = touch_button("Open")
        open_button.clicked.connect(lambda: self.open(self.run_folder.text()))
        top = QHBoxLayout()
        top.addWidget(run_row, stretch=1)
        top.addWidget(open_button)
        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.viewer = None
        self.runs = QListWidget()
        self.runs.setVisible(window.share is not None)
        self.runs.setMaximumWidth(360)
        self.runs.itemClicked.connect(self.open_item)
        self._states = {}
        view = QVBoxLayout()
        view.addLayout(top)
        view.addWidget(self.canvas, stretch=1)
        layout = QHBoxLayout(self)
        layout.addWidget(self.runs)
        layout.addLayout(view, stretch=1)
        if window.share is not None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(RESULTS_REFRESH_MS)

    def states(self):
        """Run ID -> job state, as last read from the shared folder."""
        return dict(self._states)

    @reported
    def refresh(self):
        """Show the run states last read by the mirror thread (the GUI never
        reads the share itself); open a run of this session once it is done."""
        runs = self.window.mirror_thread.runs
        old = self._states
        self._states = {run["name"]: run["state"] for run in runs}
        self.runs.clear()
        for run in runs:
            self.runs.addItem(stations.run_text(run))
        for name, state in self._states.items():
            fresh = state == "done" and old.get(name, "") != "done"
            if fresh and name in self.window.recorded_here:
                self.open(self.window.shared_run(name))
                self.window.tabs.setCurrentWidget(self)

    def open_item(self, item):
        name = item.text().split(": ")[0]
        if self._states.get(name) == "done":
            self.open(self.window.shared_run(name))

    def open(self, run_folder, out_folder=None):
        self.run_folder.setText(str(run_folder))
        self.figure.clear()
        self.viewer = TowViewer(run_folder, out_folder, fig=self.figure)
        self.canvas.draw_idle()


class ProcessThread(QThread):
    done = Signal(object)  # run folder, or an exception

    def __init__(self, run_folder):
        super().__init__()
        self.run_folder = run_folder

    def run(self):
        try:
            process.process_run(self.run_folder)
            self.done.emit(self.run_folder)
        except Exception as error:  # noqa: BLE001 (thread boundary: shown in the GUI)
            self.done.emit(error)


class MirrorThread(QThread):
    """Copies the spooled runs to the shared folder in the background, and
    keeps the heartbeats, so the GUI never waits for the network."""

    def __init__(self, share, spool):
        super().__init__()
        self.share, self.spool = share, spool
        # The GUI appends, this thread removes: list operations are atomic.
        self.mirrors = []
        self._stop = threading.Event()
        self.processing_alive = False
        self.runs = []  # the run list, newest first (stations.run_states)
        self.heartbeat = stations.Heartbeat(share, stations.CAPTURE_HEARTBEAT)
        self.watch = stations.Watch(share, stations.PROCESSING_HEARTBEAT)

    def run(self):
        last_beat = 0.0
        while not self._stop.is_set():
            self.processing_alive = self.watch.alive()
            try:
                self.runs = stations.run_states(self.share)
                stations.clean_spool(self.spool, self.share)
            except OSError:
                pass  # share unreachable: keep the last run list
            if time.monotonic() - last_beat >= stations.HEARTBEAT_S:
                try:
                    if self.heartbeat.path.parent.is_dir():
                        self.heartbeat.beat()
                        last_beat = time.monotonic()
                except OSError:
                    pass  # share unreachable
            for mirror in list(self.mirrors):
                try:
                    if mirror.sync():
                        self.mirrors.remove(mirror)
                except OSError:
                    pass  # share unreachable: the files stay spooled, retry

            self._stop.wait(MIRROR_INTERVAL_S)

    def stop(self):
        self._stop.set()
        self.wait(10_000)


def default_spool():
    return Path.home() / "digital_thread" / "spool"


class MainWindow(QMainWindow):
    def __init__(self, camera, store=None, share=None, spool=None):
        """`share` selects the capture role (section 8): runs are recorded into
        `spool` and handed over to the processing station through `share`."""
        super().__init__()
        self.setWindowTitle("Tow monitoring")
        self.camera = camera
        self.store = store or QSettings("UniSQ", "DigitalThread")
        self.share = Path(share) if share is not None else None
        self.spool = Path(spool) if spool is not None else default_spool()
        self.recorded_here = set()  # run IDs recorded in this session
        self._mirrors = {}
        self.mirror_thread = None
        if self.share is not None:
            self.mirror_thread = MirrorThread(self.share, self.spool)
        self.offline_banner = QLabel(
            "Processing station offline: capture works, jobs wait until it is started."
        )
        self.offline_banner.setStyleSheet("background: #c62828; color: white;")
        self.offline_banner.setMinimumHeight(40)
        self.offline_banner.hide()
        if self.share is not None:
            self.mirror_thread.start()
            self.banner_timer = QTimer(self)
            self.banner_timer.timeout.connect(self.show_liveness)
            self.banner_timer.start(BANNER_REFRESH_MS)
            self.resume_spooled_runs()
        self.tabs = QTabWidget()
        self.preview = PreviewTab(self)
        self.calibration = CalibrationTab(self)
        self.record = RecordTab(self)
        self.results = ResultsTab(self)
        for widget, name in (
            (self.preview, "Preview"),
            (self.calibration, "Calibration"),
            (self.record, "Record"),
            (self.results, "Results"),
        ):
            self.tabs.addTab(widget, name)
        self.close_button = QPushButton("Close")  # the touchscreen has no title bar
        self.close_button.clicked.connect(self.close)
        self.tabs.setCornerWidget(self.close_button)
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.offline_banner)
        column.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(central)
        self.grabber = Grabber(camera, self.preview.pin_input.currentData())
        self.grabber.frame.connect(self.on_frame)
        self.grabber.start()
        self._processing = None

    def tab_names(self):
        return [self.tabs.tabText(i) for i in range(self.tabs.count())]

    def on_frame(self, raw, timestamp_s, generation):
        try:
            if not self.grabber.current(generation):
                return  # grabbed before the last trigger switch
            if self.tabs.currentWidget() is self.preview:
                self.preview.show_frame(raw)
            self.record.on_frame(raw, timestamp_s)
            self.calibration.on_frame(raw, timestamp_s)
        finally:
            self.grabber.handled()

    def set_trigger(self, mode):
        self.grabber.set_trigger(mode)

    def release_trigger(self):
        """Back to free run, unless a recording or marker capture still needs the pin."""
        if not (self.record.active() or self.calibration.capturing()):
            self.grabber.set_trigger("free")

    def set_exposure(self, value):
        with self.grabber.lock:
            self.camera.set_exposure_us(value)

    def set_gain(self, value):
        with self.grabber.lock:
            self.camera.set_gain_db(value)

    def show_liveness(self):
        self.offline_banner.setVisible(not self.mirror_thread.processing_alive)

    def shared_run(self, run_id):
        """The run's folder in the shared folder (None without the capture role)."""
        return None if self.share is None else self.share / stations.RUNS / run_id

    def mirror(self, spool_folder, recorded_here=True):
        """Start copying a spooled run to the shared folder (once per run: a new
        reference in the same run keeps its mirror)."""
        if self.share is None or spool_folder.name in self._mirrors:
            return
        mirror = stations.Mirror(spool_folder, self.shared_run(spool_folder.name))
        self._mirrors[spool_folder.name] = mirror
        self.mirror_thread.mirrors.append(mirror)
        if recorded_here:
            self.recorded_here.add(spool_folder.name)

    def resume_spooled_runs(self):
        """Runs left in the spool by an earlier session (closed during an
        outage, or before the copy was complete) are finished and handed over."""
        if not self.spool.is_dir():
            return
        for folder in sorted(self.spool.iterdir()):
            if (folder / "run.json").exists():
                self.mirror(folder, recorded_here=False)
                self.hand_over(folder)

    def hand_over(self, spool_folder):
        """After Stop: the run becomes a job once every file is copied."""
        self._mirrors[spool_folder.name].finish()

    def process(self, run_folder):
        self.record.status.setText(self.record.status.text() + "\nProcessing ...")
        self._processing = ProcessThread(run_folder)
        self._processing.done.connect(self.processed)
        self._processing.start()

    def processed(self, result):
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Processing failed", str(result))
            return
        self.record.status.setText(self.record.status.text() + " done.")
        self.results.open(result)
        self.tabs.setCurrentWidget(self.results)

    def closeEvent(self, event):
        if self.record.active():
            QMessageBox.warning(self, "Close", "Stop the recording first.")
            event.ignore()
            return
        if self.mirror_thread is not None:
            self.mirror_thread.stop()
        self.grabber.stop()
        self.camera.close()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tow monitoring app.")
    parser.add_argument("--replay", help="run folder served by a replay camera")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="replay speed-up (1: nominal period)"
    )
    parser.add_argument(
        "--role", choices=["capture"], help="capture station (section 8)"
    )
    parser.add_argument("--share", help="shared folder (capture role)")
    parser.add_argument("--spool", help="local spool folder (capture role)")
    args = parser.parse_args(argv)
    if args.role == "capture" and not args.share:
        parser.error("--role capture needs --share")
    app = QApplication.instance() or QApplication(sys.argv)
    font = app.font()
    font.setPointSize(TOUCH_FONT)
    app.setFont(QFont(font))
    if args.replay:
        camera = ReplayCamera(args.replay, speed=args.speed, loop=True)
    else:
        camera = BaslerCamera()
    share = args.share if args.role == "capture" else None
    window = MainWindow(camera, share=share, spool=args.spool)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
