"""Camera interface (production): the Basler camera and a replay camera.

A camera grabs raw frames (uint16, 3000 x 4096) with a timestamp in seconds
from its own clock. Trigger mode "free" runs continuously (Preview);
"hardware" exposes one frame per rising edge on Line 1; "software" one frame
per `software_trigger()`, which the app calls for each pin pulse on the Pi's
GPIO26 (pin input, section 3).
"""

import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from digital_thread import dataset


class Camera(Protocol):
    polar_pattern: tuple[int, int, int, int]

    def settings(self) -> dict: ...  # {"exposure_us": ..., "gain_db": ...}

    def set_exposure_us(self, value: float) -> None: ...

    def set_gain_db(self, value: float) -> None: ...

    def set_trigger(self, mode: str) -> None: ...  # "free", "hardware", "software"
    def software_trigger(self) -> None: ...  # one frame in "software" mode

    def grab(self, timeout_s: float = 1.0) -> tuple[np.ndarray, float] | None: ...

    def close(self) -> None: ...


class ReplayCamera:
    """Serves the frames of a run folder as if the pin fired.

    `events` is a list of (frame name, timestamp_s); by default the run's frame
    log. A frame name may repeat (a pin bounce) or be left out (a missed
    pulse). With `speed`, `grab` waits the time between events divided by it
    (demos); without, frames come at once (tests). `loop` restarts at the end.
    In "software" trigger mode, each `software_trigger()` releases one frame.
    """

    def __init__(self, run_folder, events=None, speed=None, loop=False):
        self.run = dataset.read_run(Path(run_folder))
        self.events = events or [(f.name, f.timestamp_s) for f in self.run.frames]
        self.polar_pattern = self.run.polar_pattern
        self._settings = dict(self.run.camera_settings)
        self.speed, self.loop = speed, loop
        self.trigger = "hardware"
        self._next = 0
        self._offset = 0.0  # added to timestamps after each loop
        self._last = None  # (timestamp, monotonic time) of the last served frame
        self._triggers = threading.Semaphore(0)

    def settings(self):
        return dict(self._settings)

    def set_exposure_us(self, value):
        self._settings["exposure_us"] = float(value)

    def set_gain_db(self, value):
        self._settings["gain_db"] = float(value)

    def set_trigger(self, mode):
        self.trigger = mode
        self._triggers = threading.Semaphore(0)  # no trigger carried over

    def software_trigger(self):
        self._triggers.release()

    def grab(self, timeout_s=1.0):
        if self.trigger == "software" and not self._triggers.acquire(timeout=timeout_s):
            return None  # no trigger within the timeout
        if self._next == len(self.events):
            if not self.loop:
                return None
            self._offset += (
                self.events[-1][1] - self.events[0][1] + (self.run.nominal_period_s)
            )
            self._next = 0
        name, stamp = self.events[self._next]
        stamp += self._offset
        if self.speed and self._last is not None:
            due = self._last[1] + (stamp - self._last[0]) / self.speed
            wait = due - time.monotonic()
            if wait > timeout_s:  # like a camera: no pulse within the timeout
                time.sleep(timeout_s)
                return None
            time.sleep(max(0.0, wait))
        self._next += 1
        self._last = (stamp, time.monotonic())
        return dataset.read_frame(self.run, name), stamp

    def close(self):
        pass


# Polariser pattern of the Sony IMX250MZR sensor in the acA4112-30umPOL:
# (top-left, top-right, bottom-left, bottom-right) in degrees.
BASLER_POLAR_PATTERN = (90, 45, 135, 0)
TIMESTAMP_TICK_S = 1e-9  # ace USB 3.0 chunk timestamp tick


class BaslerCamera:
    """Basler acA4112-30umPOL via pypylon (Mono12, chunk timestamps).

    Trigger modes: "free" (continuous, latest image only: Preview), "hardware"
    (Line 1, rising edge), "software" (one frame per `software_trigger()`).
    """

    def __init__(self, serial=None):
        from pypylon import pylon

        self._pylon = pylon
        factory = pylon.TlFactory.GetInstance()
        devices = [
            d
            for d in factory.EnumerateDevices()
            if serial is None or d.GetSerialNumber() == serial
        ]
        if not devices:
            raise RuntimeError("no Basler camera found")
        self.cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        self.cam.Open()
        try:
            self.cam.PixelFormat.Value = "Mono12"
            self.cam.ChunkModeActive.Value = True
            self.cam.ChunkSelector.Value = "Timestamp"
            self.cam.ChunkEnable.Value = True
            self.polar_pattern = BASLER_POLAR_PATTERN
            self.trigger = None
            self.set_trigger("free")
        except Exception:
            self.cam.Close()
            raise

    @staticmethod
    def available():
        try:
            from pypylon import pylon
        except ImportError:
            return False
        return len(pylon.TlFactory.GetInstance().EnumerateDevices()) > 0

    def settings(self):
        return {
            "exposure_us": float(self.cam.ExposureTime.Value),
            "gain_db": float(self.cam.Gain.Value),
        }

    def set_exposure_us(self, value):
        self.cam.ExposureTime.Value = float(value)

    def set_gain_db(self, value):
        self.cam.Gain.Value = float(value)

    def set_trigger(self, mode):
        if mode not in ("free", "hardware", "software"):
            raise ValueError(mode)
        self.cam.StopGrabbing()
        self.cam.TriggerSelector.Value = "FrameStart"
        if mode == "free":
            self.cam.TriggerMode.Value = "Off"
            strategy = self._pylon.GrabStrategy_LatestImageOnly
        else:
            self.cam.TriggerMode.Value = "On"
            self.cam.TriggerSource.Value = "Line1" if mode == "hardware" else "Software"
            if mode == "hardware":
                self.cam.TriggerActivation.Value = "RisingEdge"
            strategy = self._pylon.GrabStrategy_OneByOne
        self.cam.StartGrabbing(strategy)
        self.trigger = mode

    def software_trigger(self):
        self.cam.WaitForFrameTriggerReady(
            1000, self._pylon.TimeoutHandling_ThrowException
        )
        self.cam.ExecuteSoftwareTrigger()

    def grab(self, timeout_s=1.0):
        result = self.cam.RetrieveResult(
            int(timeout_s * 1000), self._pylon.TimeoutHandling_Return
        )
        if not result.IsValid():  # timeout
            return None
        try:
            if not result.GrabSucceeded():
                return None
            raw = np.array(result.Array, dtype=np.uint16)
            return raw, result.ChunkTimestamp.Value * TIMESTAMP_TICK_S
        finally:
            result.Release()

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
