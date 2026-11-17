# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Anthony Charretier
import glob
import math
import sys

import numpy as np

R0, PITCH = 40.0, 30.0


def measure(path, layer):
    c = np.load(path)
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    ok = np.isfinite(x)
    dx, dy = x - (1500 + 0.3 * z), y - (1500 - 0.2 * z)
    th = np.mod(np.arctan2(dy, dx), 2 * math.pi)
    u = (np.hypot(dx, dy) - R0) / PITCH - th / (2 * math.pi)
    d = np.abs(u - np.round(u)) * PITCH
    label = np.round(u) + 1
    good = ok & (d < 2.0) & (label == layer)
    return ok, d, good


layer = int(sys.argv[2])
for path in sorted(glob.glob(sys.argv[1] + "/*_coords.npy")):
    ok, d, good = measure(path, layer)
    rows = ok.shape[0]
    print(f"{path}: {rows} slices x {ok.shape[1]} columns; sampled {ok.mean():.1%}; on the sheet of layer "
          f"{layer} (within 2 px): {good.sum() / max(ok.sum(), 1):.1%} of the sampled; "
          f"median distance {np.median(d[ok]):.2f} px")
