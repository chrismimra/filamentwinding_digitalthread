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
