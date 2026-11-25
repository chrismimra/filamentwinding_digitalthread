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
from pathlib import Path

from digital_thread import dataset, process

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
        if not self.source.exists():
            return False
        if not self.target.parent.is_dir():
            raise ShareUnreachable(f"{self.target.parent} is not reachable")
        for path, relative, key in self._pending():
            target = self.target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(path, tmp)
            tmp.replace(target)
            self._copied[relative] = key
        if self.finished and not self._pending():
            dataset.write_json(self.target / JOB_FILE, {"kind": "run"})
            self.job_written = True
        return self.job_written


def clean_spool(spool, share):
    """Remove spooled runs that the processing station has confirmed done."""
    spool = Path(spool)
    if not spool.exists():
        return
    for folder in spool.iterdir():
        if folder.is_dir() and job_state(Path(share) / RUNS / folder.name) == "done":
            shutil.rmtree(folder, ignore_errors=True)


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
            for job in (self.share / RUNS).glob(f"*/{JOB_FILE}")
            if job_state(job.parent) not in FINISHED
        ]
        return sorted(jobs, key=lambda folder: (folder / JOB_FILE).stat().st_mtime)

    def poll_once(self):
        """Handle the oldest open job; returns its folder, or None."""
        jobs = self._jobs()
        if not jobs:
            return None
        folder = jobs[0]
        name = f"run {folder.name}"
        self.log(f"Processing {name} ...")
        write_status(folder, "working")
        try:
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
