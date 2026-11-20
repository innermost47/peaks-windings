import numpy as np
import cv2
import sys


def fill_vertical(values, coords_meta, max_gap=3):
    """
    Remplit les trous verticaux courts (< max_gap pixels) en copiant
    le pixel valide le plus proche dans la même colonne.

    values      : (H, W) float32, 0 ou NaN = trou
    coords_meta : (H, W, 4) float32, on écrit 2.0 dans filled pour les pixels copiés
    """
    H, W = values.shape
    out = values.copy()
    meta = coords_meta.copy()
    filled_count = 0

    for x in range(W):
        col = out[:, x]
        valid = np.isfinite(col) & (col > 0)

        # Trouver les runs de trous entre deux pixels valides
        r = 0
        while r < H:
            if valid[r]:
                r += 1
                continue
            # début du trou
            start = r
            while r < H and not valid[r]:
                r += 1
            end = r  # premier pixel valide après le trou (ou H)
            if start == 0 or end == H:
                continue  # trou sur un bord, on ne remplit pas
            gap = end - start
            if gap > max_gap:
                continue  # trop grand, on laisse noir (fill-z aurait fait mieux)

            # Copier du voisin le plus proche (alternance haut/bas)
            for k in range(gap):
                if k % 2 == 0:
                    src = start - 1 - (k // 2) if k // 2 < start else start - 1
                else:
                    src = end + (k // 2) if end + (k // 2) < H else end
                if src < 0 or src >= H:
                    continue
                out[start + k, x] = out[src, x]
                meta[start + k, x, 0:3] = meta[src, x, 0:3]  # mêmes coords
                meta[start + k, x, 3] = 2.0  # marqueur "copie verticale"
                filled_count += 1

    return out, meta, filled_count


if __name__ == "__main__":
    base = "results/png/PHerc0125/layer_-27/unrolled_layer_-27"
    if len(sys.argv) > 1:
        base = sys.argv[1]

    values = np.load(f"{base}_values.npy")
    meta = np.load(f"{base}_coords_meta.npy")
    print(f"loaded values {values.shape}, meta {meta.shape}")
    print(
        f"avant : {np.sum(values > 0):,} pixels valides "
        f"({100 * np.mean(values > 0):.1f} %)"
    )

    for gap in [2, 3, 5, 10]:
        v2, m2, n = fill_vertical(values, meta, max_gap=gap)
        print(
            f"  max_gap={gap:>2} : {np.sum(v2 > 0):,} pixels "
            f"({100 * np.mean(v2 > 0):.1f} %)  +{n:,} remplis"
        )

    # Sauvegarder avec max_gap=5 (compromis raisonnable)
    v2, m2, _ = fill_vertical(values, meta, max_gap=5)
    np.save(f"{base}_values_vfill.npy", v2)
    np.save(f"{base}_coords_meta_vfill.npy", m2)

    # PNG pour comparaison rapide
    vmin, vmax = np.percentile(v2[v2 > 0], [2, 98])
    img = np.clip((v2 - vmin) / (vmax - vmin), 0, 1)
    cv2.imwrite(f"{base}_full_vfill.png", (img * 255).astype(np.uint8))
    print(f"\nwritten: {base}_values_vfill.npy")
    print(f"         {base}_coords_meta_vfill.npy")
    print(f"         {base}_full_vfill.png")
