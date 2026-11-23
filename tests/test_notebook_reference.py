"""Reference tests: production code gives the same results as the notebook.

The scientific code is written in the notebook. Each reference cell starts with
a marker comment; the test runs that cell on its own and compares results.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from digital_thread import forward_model, touch_down

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
