"""Calibration tab (slice 2): from images to a bundle that `process_run` accepts.

The session runs the whole calibration on the synthetic calibration images;
the GUI test drives the same steps through the tab.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from digital_thread import calibration, dataset, process
from digital_thread.app import MainWindow
from digital_thread.camera import ReplayCamera

SYNTHETIC = Path(__file__).parents[1] / "demoData" / "synthetic"
IMAGES = SYNTHETIC / "calibration_images"


@pytest.fixture(scope="module")
def truth():
    return json.loads((IMAGES / "truth.json").read_text(encoding="utf-8"))


def _run_with_bundle(bundle, tmp_path):
    """The banana run (reference and revolution 0) referring to `bundle`."""
    source = dataset.read_run(SYNTHETIC / "banana" / "run")
    folder = tmp_path / "run"
    folder.mkdir()
    shutil.copyfile(source.folder / source.tow_file, folder / source.tow_file)
    source.frames = source.frames[:2]
    for frame in source.frames:
        shutil.copyfile(source.folder / frame.file, folder / frame.file)
    source.calibration = str(bundle)
    dataset.write_run(folder, source)
    return folder


def test_session_writes_a_bundle_that_processes(truth, tmp_path):
    banana = dataset.read_calibration(IMAGES / truth["calibration"])
    session = calibration.Session()
    session.board_images = [
        dataset.read_image(f) for f in sorted(IMAGES.glob("board_*.png"))
    ]
    session.calibrate_optics()
    session.load_mandrel(IMAGES / truth["calibration"])
    session.marker_table = IMAGES / truth["markers"]["table"]
    session.marker_images = [
        dataset.read_image(f) for f in sorted(IMAGES.glob("markers_*.png"))
    ]
    assert session.markers_off_mesh() == [5, 6, 16, 17]
    session.solve_position()
    assert session.body_angle_guide() == pytest.approx(90.0, abs=1.0)
    session.touch_down_deg = banana.touch_down_deg
    session.crop_polygon = banana.crop_polygon
    session.save(tmp_path / "calibration")

    cal = dataset.read_calibration(tmp_path / "calibration")
    assert cal.pin_phase_deg == 0.0 and cal.image_size == (2048, 1500)
    np.testing.assert_allclose(cal.crop_polygon, banana.crop_polygon)
    assert (tmp_path / "calibration" / dataset.MARKER_TABLE_FILE).exists()
    rows, _ = dataset.read_export(
        process.process_run(_run_with_bundle(tmp_path / "calibration", tmp_path))
    )
    assert [r["revolution"] for r in rows] == [0]


def test_calibration_tab_runs_the_steps(cylinder_geometry, truth, tmp_path):
    app = QApplication.instance() or QApplication([])
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(ReplayCamera(cylinder_geometry[0].folder, loop=True), store)
    try:
        tab = win.calibration
        tab.load_images(IMAGES)
        tab.calibrate_optics()
        assert "Optics from 12 images" in tab.status.text()
        tab.mandrel.setText(str(IMAGES / truth["calibration"]))
        tab.load_mandrel()
        assert [b.value() for b in tab.axis_dir] == [0, 1, 0]
        tab.marker_table.setText(str(IMAGES / truth["markers"]["table"]))
        tab.camera_position[1].setValue(250)
        tab.solve_position()
        text = tab.status.text()
        assert "Not on the mesh (ignored): markers [5, 6, 16, 17]" in text
        assert "WARNING" not in text and "Body angle facing the camera" in text
        for x, y in [(300, 500), (1700, 500), (1700, 1000)]:
            tab.session.crop_polygon = (
                np.array([[x, y]])
                if tab.session.crop_polygon is None
                else np.vstack([tab.session.crop_polygon, [x, y]])
            )
        tab.touch_down.setValue(74)
        tab.bundle_folder.setText(str(tmp_path / "bundle"))
        tab.save()
        cal = dataset.read_calibration(tmp_path / "bundle")
        assert cal.touch_down_deg == 74 and len(cal.crop_polygon) == 3
    finally:
        win.close()
        app.processEvents()
