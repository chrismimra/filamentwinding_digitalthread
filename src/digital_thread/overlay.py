"""Overlay image of a revolution (Tow matching, production display code).

The flattened image of the fibre angle in grey, the measured tow coloured and
the planned tow edges (+- tow width / 2 around the planned path) drawn last.
"""

import numpy as np

from digital_thread.mapping import flat_coordinates, flatten

BACKGROUND = (255, 255, 255)
VALID_COLOUR = (190, 190, 190)
TOW_COLOUR = (230, 120, 20)
PLANNED_COLOUR = (20, 60, 200)
EDGE_SAMPLES = 10  # edge points per path step, so the drawn edge is continuous


def planned_edges(planned, k, cal, geo, tow_width_mm):
    """Flattened-image (row, col) cells of the planned tow edges of revolution k."""
    sel = planned.revolution == k
    path, normal = planned.point[sel], planned.normal[sel]
    if len(path) < 2:
        return np.empty((0, 2), dtype=int)
    tangent = np.gradient(path, axis=0)
    side = np.cross(normal, tangent)
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    fine = np.linspace(0, len(path) - 1, (len(path) - 1) * EDGE_SAMPLES + 1)
    cells = []
    for sign in (-1, 1):
        edge = path + sign * tow_width_mm / 2 * side
        edge = np.column_stack(
            [np.interp(fine, np.arange(len(edge)), edge[:, i]) for i in range(3)]
        )
        axial, arc = flat_coordinates(
            edge, cal.axis_point, cal.axis_dir, geo.r_ref, geo.phi_centre_deg
        )
        col = np.rint((axial - geo.flat_origin[0]) / geo.flat_spacing).astype(int)
        row = np.rint((arc - geo.flat_origin[1]) / geo.flat_spacing).astype(int)
        cells.append(np.column_stack([row, col]))
    cells = np.vstack(cells)
    gh, gw = geo.flat_pixel.shape
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < gh)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < gw)
    )
    return cells[inside]


def overlay_image(fibre, tow_cells, planned, k, cal, geo, tow_width_mm):
    """RGB overlay (uint8) of revolution k in the flattened image."""
    image = np.empty((*geo.flat_pixel.shape, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    image[np.isfinite(flatten(fibre, geo))] = VALID_COLOUR
    image[tow_cells] = TOW_COLOUR
    rows, cols = planned_edges(planned, k, cal, geo, tow_width_mm).T
    image[rows, cols] = PLANNED_COLOUR
    return image
