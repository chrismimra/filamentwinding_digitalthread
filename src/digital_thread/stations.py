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
"""

import json
import shutil
from pathlib import Path

from digital_thread import dataset, process

RUNS = "runs"
CALIBRATIONS = "calibrations"
JOB_FILE = "job.json"
STATUS_FILE = "status.json"
FINISHED = ("done", "failed")


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


class ProcessingStation:
    """Works through the jobs in the shared folder, one per `poll_once()`."""

    def __init__(self, share):
        self.share = Path(share)
        for name in (RUNS, CALIBRATIONS):
            (self.share / name).mkdir(parents=True, exist_ok=True)

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
        write_status(folder, "working")
        try:
            process.process_run(
                folder,
                progress=lambda done, total: write_status(
                    folder, "working", progress=[done, total]
                ),
            )
        except Exception as error:  # noqa: BLE001 (reported to the capture station)
            write_status(folder, "failed", f"{type(error).__name__}: {error}")
        else:
            write_status(folder, "done")
        return folder
