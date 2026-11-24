"""End-to-end Conversion and Mapping on the synthetic banana.

The banana's normals tilt along the axis and its body lies off the axis, so
this checks the inverse forward model beyond analytic surfaces. Expected
values come from the committed ground truth.
"""

from pathlib import Path

import numpy as np
import pytest

from digital_thread import conversion, dataset, geometry, mapping

RUN = Path(__file__).parents[1] / "demoData" / "synthetic" / "banana" / "run"


@pytest.fixture(scope="module")
def rev002():
    run = dataset.read_run(RUN)
    cal = dataset.read_calibration(run.calibration_folder)
    geo = geometry.compute(cal)
    maps = conversion.convert(dataset.read_frame(run, "rev002"), run.polar_pattern)
    fibre = mapping.fibre_angles(maps, geo, cal, run.processing)
    truth = np.load(run.folder / "rev002_truth.npz")
    return fibre, truth


def test_new_tow_gets_the_true_fibre_angle(rev002):
    fibre, truth = rev002
    new_tow = (truth["revolution"] == 2) & np.isfinite(fibre)
    assert new_tow.sum() > 1000
    error = (fibre[new_tow] - truth["fibre_deg"][new_tow] + 90) % 180 - 90
    # Signed median = bias; single pixels scatter by about 1 deg from shot noise.
    assert abs(np.median(error)) < 0.2
