# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Anthony Charretier
import json
import math
import os
import sys

import numpy as np

R0, PITCH = 40.0, 30.0
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
OFFSETS = [(0, 0), (260, 180), (-240, 210), (200, -260), (-220, -200)]
N_RAYS, Z = 180, 300


def cx(z):
    return 1500 + 0.3 * z


def cy(z):
    return 1500 - 0.2 * z


holes = (
    {}
)  # (summit, block of 20 slices) -> list of angle windows (degrees) the detector misses
for s in range(5):
    for b in range(Z // 20 + 1):
        holes[(s, b)] = [(rng.uniform(0, 360), rng.uniform(10, 25)) for _ in range(3)]
t = np.arange(0, 1500, 1.0)
for z in range(Z):
    lines, peaks = [], []
    centre_summit = (z // 50) % 5  # the centre jumps from summit to summit
    for s, (ox, oy) in enumerate(OFFSETS):
        sx, sy = cx(z) + ox, cy(z) + oy
        track = (
            100 * s + z // 60
        )  # a summit's track ends every 60 slices: the next one relays it
        peaks.append({"x": sx, "y": sy, "height": 10.0, "score": 0.05, "track": track})
        for ray in range(N_RAYS):
            phi = 2 * math.pi * ray / N_RAYS
            px, py = sx + t * math.cos(phi), sy + t * math.sin(phi)
            dx, dy = px - cx(z), py - cy(z)
            th = np.mod(np.arctan2(dy, dx), 2 * math.pi)
            u = (np.hypot(dx, dy) - R0) / PITCH - th / (2 * math.pi)
            fl = np.floor(u)
            idx = np.nonzero((np.diff(fl) != 0) & (np.abs(np.diff(u)) < 0.5))[0]
            crossings = []
            for i in idx:
                n = max(fl[i], fl[i + 1])
                w = (n - u[i]) / (u[i + 1] - u[i])
                x, y = px[i] + w * (px[i + 1] - px[i]), py[i] + w * (py[i + 1] - py[i])
                ang = math.degrees(math.atan2(y - cy(z), x - cx(z))) % 360
                if any((ang - a) % 360 < width for a, width in holes[(s, z // 20)]):
                    continue  # a hole of the detector
                pull = abs(rng.normal())
                if rng.random() < 0.03:  # an outlier: half a turn off, pulled hard
                    r = math.hypot(x - cx(z), y - cy(z)) + PITCH / 2
                    x, y = cx(z) + r * math.cos(math.radians(ang)), cy(
                        z
                    ) + r * math.sin(math.radians(ang))
                    pull = 6.0
                crossings.append(
                    {
                        "x": round(x + rng.normal(0, 0.4), 2),
                        "y": round(y + rng.normal(0, 0.4), 2),
                        "z": z,
                        "winding_label": int(n) + 1,
                        "u_continuous": float(n),
                        "fragment": 0,
                        "pull": round(pull, 2),
                    }
                )
            lines.append(
                {
                    "ray_index": ray,
                    "peak_index": s,
                    "track": track,
                    "angle": math.degrees(phi),
                    "crossings": crossings,
                }
            )
    ox, oy = OFFSETS[centre_summit]
    centre = {
        "x": cx(z) + ox,
        "y": cy(z) + oy,
        "peak_index": centre_summit,
        "track": 100 * centre_summit + z // 60,
    }
    with open(os.path.join(out, f"scroll_z{z:05d}.json"), "w") as f:
        json.dump(
            {
                "z": z,
                "n_rays": N_RAYS,
                "centre": centre,
                "peaks": peaks,
                "lines": lines,
            },
            f,
        )
print("written", Z)
