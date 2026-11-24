import os

# Tests run without a display; set before anything imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pytest

from digital_thread import dataset, geometry

SYNTHETIC = Path(__file__).parents[1] / "demoData" / "synthetic"


def _load_with_geometry(name):
    run = dataset.read_run(SYNTHETIC / name / "run")
    cal = dataset.read_calibration(run.calibration_folder)
    return run, cal, geometry.compute(cal)


@pytest.fixture(scope="session")
def cylinder_geometry():
    """Run, calibration and run geometry of the synthetic cylinder (computed once)."""
    return _load_with_geometry("cylinder")


@pytest.fixture(scope="session")
def banana_geometry():
    """Run, calibration and run geometry of the synthetic banana (computed once)."""
    return _load_with_geometry("banana")
