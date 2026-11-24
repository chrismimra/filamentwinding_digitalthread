"""Preview checks (production, slice 1): saturation warning and focus aid."""

import numpy as np
from scipy import ndimage

from digital_thread.conversion import MAX_ADU

EDGE_FRACTION = 0.25  # edge mask: |Laplacian| above this fraction of its maximum


def saturation(raw):
    """Super-pixels with any polariser channel at full scale, and their count."""
    raw = np.asarray(raw)
    full = raw >= MAX_ADU
    mask = full[0::2, 0::2] | full[0::2, 1::2] | full[1::2, 0::2] | full[1::2, 1::2]
    return mask, int(mask.sum())


def focus(s0):
    """Focus value (variance of the Laplacian; higher is sharper) and an edge mask
    for display."""
    s0 = np.nan_to_num(np.asarray(s0, dtype=float))
    laplace = ndimage.laplace(s0)
    magnitude = np.abs(laplace)
    peak = magnitude.max()
    edges = magnitude > EDGE_FRACTION * peak if peak > 0 else np.zeros(s0.shape, bool)
    return float(laplace.var()), edges
