# Conversion (section 4). Production copy of the notebook cell, kept identical
# by the reference test.

import numpy as np

from digital_thread.dataset import PolarisationMaps

MAX_ADU = 4095  # 12-bit: a channel at this value is saturated
AOLP_SIGN = 1  # -1 if the camera's polariser angles run counter-clockwise on screen


def convert(raw, polar_pattern):
    """Raw frame (2H x 2W) -> polarisation maps (H x W), one value per 2x2 super-pixel.

    polar_pattern: polariser angle of (top-left, top-right, bottom-left, bottom-right).
    DoLP and AoLP are NaN where a channel is saturated or S0 is zero.
    """
    raw = np.asarray(raw)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels = {
        angle: raw[r::2, c::2].astype(np.float64)  # float: uint16 differences wrap
        for (r, c), angle in zip(slots, polar_pattern, strict=True)
    }
    i0, i45, i90, i135 = (channels[a] for a in (0, 45, 90, 135))
    saturated = np.any([ch >= MAX_ADU for ch in channels.values()], axis=0)

    s0 = 0.5 * (i0 + i45 + i90 + i135)
    s1 = i0 - i90
    s2 = AOLP_SIGN * (i45 - i135)

    valid = (s0 > 0) & ~saturated
    dolp = np.full(s0.shape, np.nan)
    dolp[valid] = np.hypot(s1[valid], s2[valid]) / s0[valid]
    aolp = np.full(s0.shape, np.nan)
    psi = 0.5 * np.degrees(np.arctan2(s2[valid], s1[valid]))
    aolp[valid] = 90.0 - np.mod(90.0 - psi, 180.0)  # wrap to (-90, 90]
    return PolarisationMaps(
        s0.astype(np.float32), dolp.astype(np.float32), aolp.astype(np.float32)
    )
