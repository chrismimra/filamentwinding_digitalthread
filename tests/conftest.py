from pathlib import Path

import pytest

from digital_thread import dataset, geometry

CYLINDER_RUN = Path(__file__).parents[1] / "demoData" / "synthetic" / "cylinder" / "run"


@pytest.fixture(scope="session")
def cylinder_geometry():
    """Run, calibration and run geometry of the synthetic cylinder (computed once)."""
    run = dataset.read_run(CYLINDER_RUN)
    cal = dataset.read_calibration(run.calibration_folder)
    return run, cal, geometry.compute(cal)
