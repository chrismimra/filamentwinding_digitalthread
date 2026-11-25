"""Processing station program (production, section 8).

Command: `uv run python -m digital_thread.station <shared folder>`
Started once (desktop shortcut) and left alone: works through the jobs in the
shared folder, shows a log and whether the capture station is connected.
"""

import argparse
import sys
import threading
from datetime import datetime

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from digital_thread import stations

POLL_S = 2.0  # wait between job searches when there is nothing to do
WATCH_MS = 1000


class Worker(QThread):
    """Calls `poll_once()` in the background, so the heartbeat keeps beating
    while a run is processed."""

    log = Signal(str)

    def __init__(self, share):
        super().__init__()
        self.station = stations.ProcessingStation(share, log=self.log.emit)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                busy = self.station.poll_once() is not None
            except OSError as error:  # share briefly unreachable: try again
                self.log.emit(f"Shared folder: {error}")
                busy = False
            if not busy:
                self._stop.wait(POLL_S)

    def stop(self):
        self._stop.set()
        self.wait()  # a job in progress is finished first


class StationWindow(QMainWindow):
    def __init__(self, share):
        super().__init__()
        self.setWindowTitle("Processing station")
        self.capture_state = QLabel("Capture station: offline")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel(f"Shared folder: {share}"))
        layout.addWidget(self.capture_state)
        layout.addWidget(self.log, stretch=1)
        self.setCentralWidget(central)
        self.worker = Worker(share)
        self.worker.log.connect(self.add_log)
        self.heartbeat = self.worker.station.heartbeat
        self.watch = stations.Watch(share, stations.CAPTURE_HEARTBEAT)
        self._beats = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(WATCH_MS)
        self.heartbeat.beat()
        self.worker.start()
        self.add_log("Started")

    def add_log(self, text):
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S} {text}")

    def tick(self):
        self._beats += 1
        if self._beats * WATCH_MS / 1000 >= stations.HEARTBEAT_S:
            self._beats = 0
            try:
                self.heartbeat.beat()
            except OSError as error:
                self.add_log(f"Heartbeat: {error}")
        alive = self.watch.alive()
        self.capture_state.setText(
            f"Capture station: {'connected' if alive else 'offline'}"
        )

    def closeEvent(self, event):
        self.timer.stop()
        self.worker.stop()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("share", help="shared folder")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = StationWindow(args.share)
    window.resize(900, 600)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
