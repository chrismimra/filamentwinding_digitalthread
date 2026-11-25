"""Recorder (production, slice 3): triggered frames into a run folder.

Usage: `capture_reference()`, then the next frame passed to `on_frame` is the
reference; `start()`, then each frame is logged as a revolution; `stop()`.
Frames passed in any other state are ignored (e.g. while previewing).

Revolution index k = round((t - t0) / T), with t0 the first frame after
`start()` and T the nominal period. A frame less than T/2 after the previous
kept frame (or with an index already logged) is a pin bounce and is dropped; skipped indices are logged as
missed. Timestamps in the frame log are seconds from the reference frame.
Raw pixels outside the crop polygon are set to 0 before saving. The run folder
carries a copy of the calibration bundle (`calibration/`), so a later
recalibration does not change the run.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from digital_thread import dataset, geometry


@dataclass
class RecordSettings:
    run_id: str
    calibration_folder: Path
    tow_file: Path | None
    tow_width_mm: float
    nominal_period_s: float
    processing: dataset.Processing = field(default_factory=dataset.Processing)


BUNDLE_FOLDER = "calibration"  # the run's copy of its calibration bundle


class Recorder:
    def __init__(self, run_folder, settings, cal, polar_pattern, camera_settings):
        self.folder = Path(run_folder)
        self.settings = settings
        self.state = "idle"  # idle, reference, ready, recording, stopped
        self._inside = np.repeat(np.repeat(geometry.crop_mask(cal), 2, 0), 2, 1)
        self.run = dataset.Run(
            run_id=settings.run_id,
            calibration=BUNDLE_FOLDER,
            tow_file=Path(settings.tow_file).name if settings.tow_file else None,
            tow_width_mm=settings.tow_width_mm,
            polar_pattern=tuple(polar_pattern),
            camera_settings=dict(camera_settings),
            nominal_period_s=settings.nominal_period_s,
            processing=settings.processing,
        )
        self._reset()

    def _reset(self):
        self.run.frames = []
        self.run.missed_revolutions = []
        self.run.dropped_bounces_s = []
        self._t_ref = self._t0 = self._t_last = None
        self._k_last = -1

    def capture_reference(self):
        """The next frame becomes the reference (empty mandrel); a new reference
        starts the frame log again."""
        self._reset()
        self.state = "reference"

    def start(self):
        if self.state != "ready":
            raise RuntimeError("capture the reference frame before recording")
        self.state = "recording"

    def stop(self):
        self.state = "stopped"

    @property
    def frame_count(self):
        """Revolution frames recorded so far."""
        return sum(f.name != "reference" for f in self.run.frames)

    def on_frame(self, raw, timestamp_s):
        """Handle a grabbed frame; returns the frame name if it was saved."""
        if self.state == "reference":
            self.folder.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.settings.calibration_folder,
                self.folder / BUNDLE_FOLDER,
                dirs_exist_ok=True,  # a new reference in the same run
            )
            if self.settings.tow_file:
                shutil.copyfile(self.settings.tow_file, self.folder / self.run.tow_file)
            self._t_ref = timestamp_s
            self._save("reference", raw, 0.0)
            self.state = "ready"
            return "reference"
        if self.state != "recording":
            return None
        t = timestamp_s - self._t_ref
        period = self.settings.nominal_period_s
        if self._t0 is None:
            self._t0 = t
        k = round((t - self._t0) / period)
        if self._t_last is not None and (
            t - self._t_last < period / 2 or k <= self._k_last
        ):
            self.run.dropped_bounces_s.append(round(t, 6))
            dataset.write_run(self.folder, self.run)
            return None
        self.run.missed_revolutions += list(range(self._k_last + 1, k))
        self._k_last, self._t_last = k, t
        name = f"rev{k:03d}"
        self._save(name, raw, t)
        return name

    def _save(self, name, raw, t):
        raw = np.where(self._inside, raw, 0).astype(np.uint16)
        dataset.write_frame(self.folder, name, raw)
        self.run.frames.append(dataset.Frame(name, dataset.frame_file(name), t))
        dataset.write_run(self.folder, self.run)
