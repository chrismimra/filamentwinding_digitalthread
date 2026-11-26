"""STEP import (slice 2): the mandrel mesh keeps only the outer surface."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh", reason="gmsh runs on the processing station only")

from digital_thread.step_import import step_to_mesh

STEP = Path(__file__).parents[1] / "demoData" / "Banana 1.STEP"


def test_joint_between_banana_segments_is_dropped():
    # The banana is built from segments; the solid middle ones touch at y = 300.
    keep = lambda name, bbox: "Banana-part" in name
    vertices, faces, normals = step_to_mesh(STEP, keep)
    centres = vertices[faces].mean(axis=1)
    face_normals = normals[faces].mean(axis=1)
    joint = (np.abs(centres[:, 1] - 300) < 0.01) & (np.abs(face_normals[:, 1]) > 0.99)
    assert not joint.any()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert centres[:, 2].min() < -195  # the bulge bottom (z = -200) is there
