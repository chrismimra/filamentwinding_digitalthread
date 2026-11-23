from pathlib import Path

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
