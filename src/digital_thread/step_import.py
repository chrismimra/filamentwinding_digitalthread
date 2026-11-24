"""STEP import (production, slice 2): mandrel STEP file -> triangle mesh.

The mesh keeps the STEP frame (mm). Vertices are duplicated per CAD surface so
each carries the exact, outward CAD normal of its surface. Volumes are fused
into one shape first (gmsh fragment), and faces between two volumes (e.g. the
joints of a mandrel built from segments) are dropped, so only the outer
surface remains.
"""

import gmsh
import numpy as np


def step_to_mesh(step_file, keep=lambda name, bbox: True, mesh_size=4.0):
    """(vertices, faces, normals) of the outer surface of the kept volumes.

    `keep(name, bbox)` selects volumes (bbox = xmin, ymin, zmin, xmax, ymax, zmax).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()
        dropped = [
            (3, tag)
            for _, tag in gmsh.model.getEntities(3)
            if not keep(
                gmsh.model.getEntityName(3, tag), gmsh.model.getBoundingBox(3, tag)
            )
        ]
        if dropped:
            gmsh.model.occ.remove(dropped, recursive=True)
            gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) > 1:  # shared faces between touching volumes
            gmsh.model.occ.fragment(volumes[:1], volumes[1:])
            gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1.5)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(2)

        vertices, faces, normals = [], [], []
        offset = 0
        for _, surface in gmsh.model.getEntities(2):
            volumes, _ = gmsh.model.getAdjacencies(2, surface)
            if len(volumes) > 1:  # interior face between two volumes
                continue
            node_tags, coords, _ = gmsh.model.mesh.getNodes(
                2, surface, includeBoundary=True
            )
            if len(node_tags) == 0:
                continue
            coords = coords.reshape(-1, 3)
            params = gmsh.model.getParametrization(2, surface, coords.ravel())
            n = np.asarray(gmsh.model.getNormal(surface, params)).reshape(-1, 3)
            probe = coords[0] + 0.01 * n[0]
            if len(volumes) and gmsh.model.isInside(3, volumes[0], probe.tolist()):
                n = -n
            index = {tag: i + offset for i, tag in enumerate(node_tags)}
            types, _, elem_nodes = gmsh.model.mesh.getElements(2, surface)
            for elem_type, nodes in zip(types, elem_nodes, strict=True):
                if elem_type == 2:  # 3-node triangle
                    faces.append(np.vectorize(index.get)(nodes.reshape(-1, 3)))
            vertices.append(coords)
            normals.append(n)
            offset += len(node_tags)
    finally:
        gmsh.finalize()
    return np.vstack(vertices), np.vstack(faces).astype(np.int64), np.vstack(normals)
