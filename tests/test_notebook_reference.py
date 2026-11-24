"""Reference tests: production code gives the same results as the notebook.

The scientific code is written in the notebook. Each reference cell starts with
a marker comment; the test runs that cell on its own and compares results.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from digital_thread import (
    conversion,
    dataset,
    forward_model,
    geometry,
    mapping,
    touch_down,
    tow_matching,
)

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_DigitalThreadNotebook.ipynb"


def notebook_cell(marker):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(marker):
            namespace = {}
            # Running the repo's own notebook cell is the point of this test.
            exec(compile(source, f"notebook:{marker}", "exec"), namespace)  # noqa: S102
            return namespace
    pytest.fail(f"no notebook code cell starts with {marker!r}")


def test_forward_model_matches_notebook():
    nb = notebook_cell("# Forward model")
    rng = np.random.default_rng(1)
    points = rng.uniform(-100, 100, (200, 3)) + [0, 0, 600]
    normals = rng.normal(size=(200, 3)) + [0, 0, -2]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    axis = np.array([0.0, 1.0, 0.0])
    rot, t = np.eye(3), np.zeros(3)
    fibre = rng.uniform(-90, 90, 200)

    for name in ("aolp_from_fibre", "fibre_from_aolp"):
        args = (fibre, points, normals, axis, rot, t)
        np.testing.assert_allclose(
            getattr(forward_model, name)(*args), nb[name](*args), atol=1e-12
        )
    args = (1000.0, 0.3, fibre, 45.0)
    np.testing.assert_allclose(
        forward_model.polariser_intensity(*args), nb["polariser_intensity"](*args)
    )
    np.testing.assert_allclose(
        forward_model.s0(0.4, normals[:, 2], 3000.0),
        nb["s0"](0.4, normals[:, 2], 3000.0),
    )
    assert forward_model.DOLP == nb["DOLP"]
    assert forward_model.ALBEDO == nb["ALBEDO"]


def test_conversion_matches_notebook():
    nb = notebook_cell("# Conversion")
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 4096, (40, 60)).astype(np.uint16)  # includes saturated
    pattern = (90, 45, 135, 0)
    ours, theirs = conversion.convert(raw, pattern), nb["convert"](raw, pattern)
    for name in ("s0", "dolp", "aolp"):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name))
    assert nb["AOLP_SIGN"] == conversion.AOLP_SIGN


def test_mapping_matches_notebook():
    nb = notebook_cell("# Mapping")
    rng = np.random.default_rng(5)
    h, w = 30, 40
    point = rng.uniform(-50, 50, (h, w, 3)) + [0, 0, 600]
    normal = rng.normal(size=(h, w, 3)) + [0, 0, -3]
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    point[:3] = normal[:3] = np.nan  # outside the crop
    geo = SimpleNamespace(
        point=point.astype(np.float32), normal=normal.astype(np.float32)
    )
    maps = dataset.PolarisationMaps(
        s0=np.full((h, w), 1000, np.float32),
        dolp=rng.uniform(0, 0.4, (h, w)).astype(np.float32),
        aolp=rng.uniform(-90, 90, (h, w)).astype(np.float32),
    )
    cal = SimpleNamespace(R=np.eye(3), t=np.zeros(3), axis_dir=np.array([0, 1.0, 0]))
    processing = dataset.Processing(dolp_threshold=0.1, incidence_limit_deg=30.0)
    for name in ("valid_pixels", "fibre_angles"):
        args = (maps, geo, cal, processing)
        np.testing.assert_array_equal(getattr(mapping, name)(*args), nb[name](*args))

    for axis in ([0, 0, 1.0], [1.0, 0, 0], [0.3, -0.5, 0.8]):
        args = (point.reshape(-1, 3), np.array([1.0, 2, 3]), np.array(axis), 80.0, 170)
        for ours, theirs in zip(
            mapping.flat_coordinates(*args), nb["flat_coordinates"](*args), strict=True
        ):
            np.testing.assert_array_equal(ours, theirs)
    geo.flat_pixel = rng.integers(-1, h * w, (7, 9)).astype(np.int32)
    np.testing.assert_array_equal(
        mapping.flatten(maps.aolp, geo), nb["flatten"](maps.aolp, geo)
    )


def test_touch_down_model_matches_notebook():
    nb = notebook_cell("# Touch-down model")
    rng = np.random.default_rng(2)
    angles = np.radians(np.cumsum(-rng.uniform(0, 5, 500)))
    points = np.column_stack(
        [100 * np.cos(angles), rng.uniform(0, 500, 500), 100 * np.sin(angles)]
    )
    args = (points, np.zeros(3), np.array([0.0, -1.0, 0.0]), 37.0, 12.0)
    np.testing.assert_array_equal(
        touch_down.revolution_index(*args), nb["revolution_index"](*args)
    )
    np.testing.assert_allclose(
        touch_down.mandrel_angle(*args), nb["mandrel_angle"](*args)
    )


def test_tow_matching_matches_notebook(cylinder_geometry):
    nb = notebook_cell("# Tow matching")
    run, cal, geo = cylinder_geometry
    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    fibre = {
        f.name: mapping.fibre_angles(
            conversion.convert(dataset.read_frame(run, f.name), run.polar_pattern),
            geo,
            cal,
            run.processing,
        )
        for f in run.frames
    }
    ours = tow_matching.tow_table(run, cal, geo, planned, fibre)
    theirs = nb["tow_table"](run, cal, geo, planned, fibre)
    assert ours == theirs
    np.testing.assert_array_equal(
        tow_matching.measured_tow(fibre["rev001"], fibre["rev002"], geo),
        nb["measured_tow"](fibre["rev001"], fibre["rev002"], geo),
    )
