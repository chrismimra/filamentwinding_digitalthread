import json
from pathlib import Path

import numpy as np
import pytest

from digital_thread import dataset

DEMO = Path(__file__).parents[1] / "demoData"


def test_reads_the_tow_placement_file_format():
    path = dataset.read_path(DEMO / "planned_Path.csv")
    assert path.shape == (1001, 3)
    assert path[0] == pytest.approx([22.823869, -51.547541, -76.2203232])


def test_path_round_trip(tmp_path):
    path = dataset.read_path(DEMO / "planned_Path.csv")
    dataset.write_path(tmp_path / "copy.csv", path)
    assert dataset.read_path(tmp_path / "copy.csv") == pytest.approx(path)


def test_polarisation_maps_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    s0, dolp, aolp = rng.uniform(0, 1, (3, 5, 7)).astype(np.float32)
    dolp[0, 0] = aolp[0, 0] = np.nan
    maps = dataset.PolarisationMaps(s0, dolp, aolp)
    dataset.write_polarisation(tmp_path, "rev000", maps)
    assert (tmp_path / "rev000_polarisation.npz").exists()
    back = dataset.read_polarisation(tmp_path, "rev000")
    for name, expected in (("s0", s0), ("dolp", dolp), ("aolp", aolp)):
        np.testing.assert_array_equal(getattr(back, name), expected)
        assert getattr(back, name).dtype == np.float32


SYNTHETIC_RUN = DEMO / "synthetic" / "cylinder" / "run"


def test_run_without_processing_settings_loads_the_defaults(tmp_path):
    meta = json.loads((SYNTHETIC_RUN / "run.json").read_text(encoding="utf-8"))
    meta.pop("processing", None)
    (tmp_path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    old = dataset.read_run(tmp_path)
    assert old.processing.dolp_threshold == 0.1
    assert old.processing.incidence_limit_deg == 20.0


def test_processing_settings_round_trip(tmp_path):
    run = dataset.read_run(SYNTHETIC_RUN)
    run.processing = dataset.Processing(dolp_threshold=0.15, incidence_limit_deg=25.0)
    dataset.write_run(tmp_path, run)
    assert dataset.read_run(tmp_path).processing == run.processing


def test_crop_polygon_round_trip_and_default(tmp_path):
    cal = dataset.read_calibration(SYNTHETIC_RUN.parent / "calibration")
    cal.crop_polygon = np.array([[10.0, 20.0], [500.0, 20.0], [300.0, 400.0]])
    dataset.write_calibration(tmp_path / "a", cal)
    back = dataset.read_calibration(tmp_path / "a")
    np.testing.assert_array_equal(back.crop_polygon, cal.crop_polygon)

    cal.crop_polygon = None
    dataset.write_calibration(tmp_path / "b", cal)
    assert dataset.read_calibration(tmp_path / "b").crop_polygon is None
