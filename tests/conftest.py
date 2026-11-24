import os

# Tests run without a display; set before anything imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Qt without a window

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


@pytest.fixture(scope="session")
def processed_committed(tmp_path_factory, request):
    """Process a committed synthetic run once per session: name -> (out, export)."""
    from digital_thread import process

    cache = {}

    def get(name):
        if name not in cache:
            run, _, geo = request.getfixturevalue(f"{name}_geometry")
            out = tmp_path_factory.mktemp(f"processed_{name}")
            dataset.write_geometry(out, geo)
            cache[name] = out, process.process_run(run.folder, out)
        return cache[name]

    return get
