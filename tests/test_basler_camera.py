"""Basler camera (slice 1): run only with a camera connected (marker `camera`)."""

import pytest

from digital_thread.camera import BaslerCamera

pytestmark = [
    pytest.mark.camera,
    pytest.mark.skipif(not BaslerCamera.available(), reason="no Basler camera"),
]


@pytest.fixture(scope="module")
def cam():
    camera = BaslerCamera()
    yield camera
    camera.close()


def test_polariser_pattern(cam):
    assert sorted(cam.polar_pattern) == [0, 45, 90, 135]


def test_exposure_and_gain_round_trip(cam):
    cam.set_exposure_us(5000)
    cam.set_gain_db(2.0)
    settings = cam.settings()
    assert settings["exposure_us"] == pytest.approx(5000, rel=0.01)
    assert settings["gain_db"] == pytest.approx(2.0, abs=0.1)


def test_software_triggered_frames_with_increasing_timestamps(cam):
    cam.set_trigger("software")
    stamps = []
    for _ in range(3):
        cam.software_trigger()
        raw, stamp = cam.grab(timeout_s=2.0)
        assert raw.shape == (3000, 4096) and raw.dtype.name == "uint16"
        assert raw.max() <= 4095
        stamps.append(stamp)
    assert stamps[0] < stamps[1] < stamps[2]
    cam.set_trigger("free")
