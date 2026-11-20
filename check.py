import numpy as np

c = np.load("results/png/PHerc0125/layer_-27/unrolled_layer_-27_coords.npy")
print(c.shape)  # (608, ~14990, 3)
z = c[..., 2]

assert np.nanmin(z) >= 4000 - 60
assert np.nanmax(z) <= 4607 + 60
r = 300
line_z = z[r]
print(
    "ligne",
    r,
    "-> z min/max/unique:",
    np.nanmin(line_z),
    np.nanmax(line_z),
    "nb valeurs distinctes:",
    len(np.unique(line_z[~np.isnan(line_z)])),
)
