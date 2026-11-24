"""Camera interface (production): the Basler camera and a replay camera.

A camera grabs raw frames (uint16, 3000 x 4096) with a timestamp in seconds
from its own clock. Trigger mode "free" runs continuously (Preview);
"hardware" exposes one frame per rising edge on Line 1 (the pin).
"""

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

    def set_trigger(self, mode: str) -> None: ...  # "free" or "hardware"

    def grab(self, timeout_s: float = 1.0) -> tuple[np.ndarray, float] | None: ...

    def close(self) -> None: ...


class ReplayCamera:
    """Serves the frames of a run folder as if the pin fired.

    `events` is a list of (frame name, timestamp_s); by default the run's frame
    log. A frame name may repeat (a pin bounce) or be left out (a missed
    pulse). With `speed`, `grab` waits the time between events divided by it
    (demos); without, frames come at once (tests). `loop` restarts at the end.
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
        self._last = None

    def settings(self):
        return dict(self._settings)

    def set_exposure_us(self, value):
        self._settings["exposure_us"] = float(value)

    def set_gain_db(self, value):
        self._settings["gain_db"] = float(value)

    def set_trigger(self, mode):
        self.trigger = mode

    def grab(self, timeout_s=1.0):
        if self._next == len(self.events):
            if not self.loop:
                return None
            self._offset += (
                self.events[-1][1] - self.events[0][1] + (self.run.nominal_period_s)
            )
            self._next = 0
        name, stamp = self.events[self._next]
        self._next += 1
        if self.speed and self._last is not None:
            time.sleep(max(0.0, (stamp - self._last) / self.speed))
        self._last = stamp
        return dataset.read_frame(self.run, name), stamp + self._offset

    def close(self):
        pass
