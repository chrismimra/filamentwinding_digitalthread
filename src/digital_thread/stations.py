"""Capture and processing stations (production, section 8).

The stations exchange jobs as files in one shared folder:

    <share>/runs/<run id>/          a run folder, copied from the capture station
    <share>/calibrations/<name>/    a calibration job folder

A job folder is complete when the capture station has written `job.json`
(its last file). The processing station writes `status.json` in the same
folder; the capture station only reads it.

The processing station creates `runs/` and `calibrations/` when it starts. The
capture station writes only into a share that holds them: an unmounted share
is an empty local folder, and frames written there would never arrive.

Each station rewrites its heartbeat file in the share root every few seconds.
The clocks of the two stations may differ, so a station is alive when its
heartbeat has *changed* within the last `OFFLINE_S` by the reader's clock.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from digital_thread import calibration, dataset, process

RUNS = "runs"
CALIBRATIONS = "calibrations"
JOB_FILE = "job.json"
STATUS_FILE = "status.json"
FINISHED = ("done", "failed")
PROCESSING_HEARTBEAT = "processing_station.json"
CAPTURE_HEARTBEAT = "capture_station.json"
HEARTBEAT_S = 5.0
OFFLINE_S = 15.0


def read_json(path):
    """The JSON content, or None if the file is missing or being replaced."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def job_state(folder):
    """copying (no job.json yet), waiting, working, done or failed."""
    folder = Path(folder)
    status = read_json(folder / STATUS_FILE)
    if status is not None:
        return status["state"]
    return "waiting" if (folder / JOB_FILE).exists() else "copying"


class ShareUnreachable(OSError):
    """The shared folder is not mounted (or the processing station never ran)."""


def write_status(folder, state, message="", progress=None):
    dataset.write_json(
        Path(folder) / STATUS_FILE,
        {"state": state, "message": message, "progress": progress},
    )


class Mirror:
    """Copies a run folder from the capture station's spool to the share.

    `sync()` copies every file that is new or changed since the last copy,
    `run.json` last, so the copy in the share is always a consistent run
    folder. After `finish()`, the first complete sync writes `job.json`.
    """

    def __init__(self, source, target):
        self.source, self.target = Path(source), Path(target)
        self._copied = {}  # relative path -> (size, mtime) of the copied version
        self.finished = False
        self.job_written = False

    def finish(self):
        self.finished = True

    def _pending(self):
        files = [
            p
            for p in self.source.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        ]
        files.sort(key=lambda p: p.name == "run.json")  # run.json last
        pending = []
        for path in files:
            stat = path.stat()
            key = (stat.st_size, stat.st_mtime_ns)
            relative = path.relative_to(self.source)
            if self._copied.get(relative) != key:
                pending.append((path, relative, key))
        return pending

    def sync(self):
        """Copy what changed; returns True once job.json is written."""
        if self.job_written:
            return True
        if self.finished and (self.target / JOB_FILE).exists():
            self.job_written = True  # handed over by an earlier session
            return True
        if not self.source.exists():
            return False
        if not self.target.parent.is_dir():
            raise ShareUnreachable(f"{self.target.parent} is not reachable")
        for path, relative, key in self._pending():
            target = self.target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(path, tmp)
            dataset.replace(tmp, target)
            self._copied[relative] = key
        if self.finished and not self._pending():
            dataset.write_json(self.target / JOB_FILE, {"kind": "run"})
            self.job_written = True
        return self.job_written


def run_states(share):
    """The runs in the share, newest first (run IDs start with the date):
    dicts with name, state, progress and message."""
    runs = Path(share) / RUNS
    if not runs.is_dir():
        return []
    states = []
    for folder in sorted((f for f in runs.iterdir() if f.is_dir()), reverse=True):
        status = read_json(folder / STATUS_FILE) or {}
        states.append(
            {
                "name": folder.name,
                "state": job_state(folder),
                "progress": status.get("progress"),
                "message": status.get("message", ""),
            }
        )
    return states


def run_text(run):
    """One line of the run list as the operator reads it."""
    state = "processing" if run["state"] == "working" else run["state"]
    text = f"{run['name']}: {state}"
    if run["state"] == "working" and run["progress"]:
        text += " {}/{}".format(*run["progress"])
    if run["state"] == "failed":
        text += f"\n{run['message']}"
    return text


def clean_spool(spool, share):
    """Remove spooled runs that the processing station has confirmed done."""
    spool = Path(spool)
    if not spool.exists():
        return
    for folder in spool.iterdir():
        if folder.is_dir() and job_state(Path(share) / RUNS / folder.name) == "done":
            shutil.rmtree(folder, ignore_errors=True)


# Calibration jobs. An optics job holds the board images; a position job holds
# the intrinsics, the mandrel (STEP file or bundle), axis, marker table and
# marker images. The processing station answers with `intrinsics.json`, or with
# a solved bundle (`solved/`, pose and mesh, touch-down 0) and `pose.json`.
INTRINSICS_FILE = "intrinsics.json"
POSE_FILE = "pose.json"
SOLVED_FOLDER = "solved"
STEP_FILE = "mandrel.step"
MANDREL_BUNDLE = "mandrel"


def _job_folder(share, kind):
    name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{kind}"
    folder = Path(share) / CALIBRATIONS / name
    if not folder.parent.is_dir():
        raise ShareUnreachable(f"{folder.parent} is not reachable")
    folder.mkdir()
    return folder


def _write_images(folder, prefix, images):
    for i, image in enumerate(images):
        dataset.write_image(folder / f"{prefix}_{i:02d}.png", image)


def _read_images(folder, prefix):
    return [dataset.read_image(f) for f in sorted(folder.glob(f"{prefix}_*.png"))]


def _intrinsics_json(intrinsics):
    return {
        "K": np.asarray(intrinsics.K).tolist(),
        "distortion": np.asarray(intrinsics.distortion).tolist(),
        "rms_px": float(intrinsics.rms_px),
        "used": int(intrinsics.used),
        "focal_from_lens": bool(intrinsics.focal_from_lens),
        "centre_fixed": bool(intrinsics.centre_fixed),
    }


def _intrinsics(data):
    return calibration.Intrinsics(
        np.array(data["K"]),
        np.array(data["distortion"]),
        data["rms_px"],
        data["used"],
        data.get("focal_from_lens", False),
        data.get("centre_fixed", False),
    )


def submit_optics(share, board_images):
    """An optics job with the board images; returns its folder."""
    folder = _job_folder(share, "optics")
    _write_images(folder, "board", board_images)
    dataset.write_json(folder / JOB_FILE, {"kind": "optics"})
    return folder


def solve_optics(folder):
    session = calibration.Session(board_images=_read_images(folder, "board"))
    dataset.write_json(
        folder / INTRINSICS_FILE, _intrinsics_json(session.calibrate_optics())
    )


def read_optics(folder):
    return _intrinsics(read_json(Path(folder) / INTRINSICS_FILE))


def submit_position(share, session, mandrel, axis_point, axis_dir):
    """A position job from the Calibration tab's session; `mandrel` is a STEP
    file or a bundle folder. Returns the job folder."""
    session.require("intrinsics", "marker_table", "marker_images")
    folder = _job_folder(share, "position")
    mandrel = Path(mandrel)
    if mandrel.is_dir():
        for name in ("calibration.json", "mesh.npz"):
            target = folder / MANDREL_BUNDLE / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(mandrel / name, target)
    else:
        shutil.copyfile(mandrel, folder / STEP_FILE)
    shutil.copyfile(session.marker_table, folder / dataset.MARKER_TABLE_FILE)
    _write_images(folder, "markers", session.marker_images)
    dataset.write_json(folder / INTRINSICS_FILE, _intrinsics_json(session.intrinsics))
    dataset.write_json(
        folder / JOB_FILE,
        {
            "kind": "position",
            "axis_point": [float(v) for v in axis_point],
            "axis_dir": [float(v) for v in axis_dir],
            "marker_size_mm": float(session.marker_size_mm),
        },
    )
    return folder


def solve_position(folder):
    job = read_json(folder / JOB_FILE)
    session = calibration.Session(
        intrinsics=_intrinsics(read_json(folder / INTRINSICS_FILE)),
        marker_table=folder / dataset.MARKER_TABLE_FILE,
        marker_size_mm=job["marker_size_mm"],
        marker_images=_read_images(folder, "markers"),
    )
    bundle = folder / MANDREL_BUNDLE
    session.load_mandrel(
        bundle if bundle.is_dir() else folder / STEP_FILE,
        job["axis_point"],
        job["axis_dir"],
    )
    pose = session.solve_position()
    dataset.write_calibration(folder / SOLVED_FOLDER, session.calibration_so_far())
    dataset.write_json(
        folder / POSE_FILE,
        {
            "centre": np.asarray(pose.centre).tolist(),
            "marker_errors_px": {str(k): v for k, v in pose.marker_errors_px.items()},
            "rms_px": float(pose.rms_px),
            "used": int(pose.used),
        },
    )


def read_position(folder, session):
    """Mesh, axis and pose of a solved position job into `session`."""
    folder = Path(folder)
    solved = dataset.read_calibration(folder / SOLVED_FOLDER)
    session.load_mandrel(folder / SOLVED_FOLDER)
    data = read_json(folder / POSE_FILE)
    session.pose = calibration.Pose(
        R=solved.R,
        t=solved.t,
        centre=np.array(data["centre"]),
        marker_errors_px={int(k): v for k, v in data["marker_errors_px"].items()},
        rms_px=data["rms_px"],
        used=data["used"],
    )
    return session.pose


class Heartbeat:
    """Writes one station's heartbeat file (a counter that changes each beat)."""

    def __init__(self, share, name):
        self.path = Path(share) / name
        self.count = 0

    def beat(self):
        self.count += 1
        dataset.write_json(self.path, {"beat": self.count, "time": time.time()})


class Watch:
    """Whether the other station is alive: its heartbeat changed recently."""

    def __init__(self, share, name, offline_s=OFFLINE_S):
        self.path = Path(share) / name
        self.offline_s = offline_s
        self._content = None
        self._changed = None  # monotonic time of the last change seen

    def alive(self):
        try:
            content = self.path.read_bytes()
        except OSError:
            content = None
        now = time.monotonic()
        if None not in (content, self._content) and content != self._content:
            self._changed = now
        self._content = content  # the first read is only a baseline
        return self._changed is not None and now - self._changed < self.offline_s


class ProcessingStation:
    """Works through the jobs in the shared folder, one per `poll_once()`.
    `log(text)` receives a line per job step."""

    def __init__(self, share, log=None):
        self.share = Path(share)
        self.log = log or (lambda text: None)
        for name in (RUNS, CALIBRATIONS):
            (self.share / name).mkdir(parents=True, exist_ok=True)
        self.heartbeat = Heartbeat(self.share, PROCESSING_HEARTBEAT)

    def _jobs(self):
        jobs = [
            job.parent
            for kind in (RUNS, CALIBRATIONS)
            for job in (self.share / kind).glob(f"*/{JOB_FILE}")
            if job_state(job.parent) not in FINISHED
        ]
        return sorted(jobs, key=lambda folder: (folder / JOB_FILE).stat().st_mtime)

    def poll_once(self):
        """Handle the oldest open job; returns its folder, or None."""
        jobs = self._jobs()
        if not jobs:
            return None
        folder = jobs[0]
        kind = (read_json(folder / JOB_FILE) or {}).get("kind", "run")
        name = f"{kind} {folder.name}"
        self.log(f"Working on {name} ...")
        write_status(folder, "working")
        try:
            if kind == "optics":
                solve_optics(folder)
            elif kind == "position":
                solve_position(folder)
            else:
                process.process_run(
                    folder,
                    progress=lambda done, total: write_status(
                        folder, "working", progress=[done, total]
                    ),
                )
        except Exception as error:  # noqa: BLE001 (reported to the capture station)
            message = f"{type(error).__name__}: {error}"
            write_status(folder, "failed", message)
            self.log(f"{name} failed: {message}")
        else:
            write_status(folder, "done")
            self.log(f"{name} done")
        return folder
