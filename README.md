# peaks_windings (C++)

C++ implementation of the layered volume segmentation, unrolling, and 3D meshing pipeline.

| Tool             | Purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `peaks_windings` | Computes consistent winding labels across Z-slices (`<name>_z<z>.json`) |
| `unroll_layer`   | Flattens a single winding layer into a 2D image (slab MIP)              |
| `export_3d`      | Exports layer geometry as PLY point clouds and OBJ meshes               |
| `overlay_sections` | Draws an unrolled layer back onto raw CT slices to check it           |

## Build

Requires OpenCV (`core`, `imgproc`, `imgcodecs`), libcurl, c-blosc, and nlohmann-json (`zstd` and `zlib` are optional).

### Windows (VS 2022 + vcpkg)

```bat
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:/vcpkg/vcpkg install --triplet x64-windows
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### Linux

```sh
sudo apt install build-essential cmake ninja-build libopencv-dev libcurl4-openssl-dev \
                 libblosc-dev libzstd-dev zlib1g-dev nlohmann-json3-dev
cmake -S . -B build -G Ninja
cmake --build build
```

Pass `-DPW_NATIVE=ON` to enable `-march=native`.

## Output layout

Every tool takes a dataset name (`--scroll-name NAME`, default `scroll`). Give the same name to every step: the
outputs are then sorted automatically, and each tool finds the files of the previous one.

```
results/
├── json/NAME/                  peaks_windings: NAME_z<z>.json, one file per slice
├── png/NAME/layer_<N>/         unroll_layer: images and pixel coordinates of layer N
│   └── sections/               overlay_sections: the unrolled layer drawn on raw slices
└── 3D/
    ├── ply/                    export_3d: NAME_pointcloud.ply
    └── obj/                    export_3d: scroll_NAME.obj
```

Folders are created as needed. `--output-dir` (`peaks_windings`), `--results-dir` and `--output` (`unroll_layer`),
`--results-dir` (`export_3d`) and `--output` (`overlay_sections`) override the default paths.

## Usage

### Recorded volumes (`scroll_settings.json`)

Each scroll keeps the volumes it was made from in `results/json/NAME/scroll_settings.json`, so they are given once:

```sh
peaks_windings --scroll-name dataset --scroll-url D:/data/surface.zarr --ct <ct_zarr_path_or_url> --z-start 0
unroll_layer --scroll-name dataset --layer N
overlay_sections --scroll-name dataset --coords results/png/dataset/layer_N/unrolled_layer_N_coords.npy
```

* `surface`: the volume `peaks_windings` read, with its shape: the voxels of every result file. Written at each run;
  `--scroll-url` defaults to it. A volume of another shape is refused for the same results (use another name).
* `ct`: the CT volume sampled by `unroll_layer` and drawn by `overlay_sections`. Given by `peaks_windings --ct` or by
  `--scroll` of either tool (the last one given is kept); `--scroll` defaults to it. `unroll_layer` requires its shape
  to be the shape of `surface`.

### Label extraction (`peaks_windings`)

```sh
peaks_windings --scroll-name dataset --scroll-url <zarr_path_or_url> --z-start 8000 --z-count 100
peaks_windings --scroll-name dataset --scroll-url D:/data/volume.zarr --z-start 0 --z-count 50 --every 2
peaks_windings --image slice.png
peaks_windings --synthetic
```

Options:

* `--scroll-url URL`: Zarr store to read: local directory, `http(s)://` or anonymous `s3://`. A multiscale store
  is read at its root, or at `0/` when the root is a group (append `/2`, etc. to the URL to pick another level).
  Default: the surface volume recorded for the scroll.
* `--ct URL`: The CT volume of the same scroll, at the same resolution, recorded for `unroll_layer` and
  `overlay_sections`.
* `--scroll-name NAME`: Dataset name, used in file names and folders (default `scroll`).
* `--output-dir DIR`: Output folder (default `results/json/NAME/`).
* `--z-start N`, `--z-count N`, `--every N`: Slices processed (stops at the end of the volume).
* `--threads N`: Worker threads (`0` = all available).
* `--centre-jobs N`: Concurrent candidate centres evaluated (default `4`; reduce to lower RAM usage).
* `--connections N`: Parallel HTTP downloads, or parallel file reads of a local store (default `32`).
* `--cache-gb F`: Memory budget for decoded chunk planes (default `2` GiB).
* `--image FILE`: Process a single image instead of a Zarr store.
* `--timings`: Log per-step execution times.

See `--help` for the detection parameters (`--peaks`, `--rays`, `--track-radius`, `--track-memory`, `--scale`, ...).

### Unrolling (`unroll_layer`)

```sh
unroll_layer --scroll-name dataset [--scroll <zarr_path_or_url>] [--layer N | --layer N,M,... | --layer all]
```

The JSON results are read from `results/json/NAME/` (`--results-dir` to read elsewhere). Without `--layer`, the
layer with the most points is unrolled; a list unrolls each layer given, `all` every layer of at least
`--min-points` points (default `1000`), densest first. The labels are aligned once and the CT volume opened once for
all of them; a layer with nothing to unroll is skipped. Without `--scroll`, the recorded CT volume is sampled.

> **Coordinates:** the CT volume must have the same shape as the volume `peaks_windings` read: the points of the
> JSON files are in voxels of that volume. When the CT store given has another shape, `unroll_layer` takes the level
> of the same multiscale store (`URL/0`, `URL/1`...) with the recorded surface shape and records it; it stops only
> when no level has that shape.

The layer is followed along its own sheet, slice after slice, from the sheet traces written by `peaks_windings`
(result files written before the traces existed cannot be unrolled: run `peaks_windings` again). Every column of the
image is one position along the sheet, the same in every slice. Outputs in `results/png/NAME/layer_<N>/`:

* `unrolled_layer_<N>.png`: annotated preview (position along the sheet, slice Z);
* `unrolled_layer_<N>_full.png`: 8-bit unrolled image, one row per slice;
* `unrolled_layer_<N>_coords.npy`: float32 `(rows, cols, 3)` volume coordinates `(x, y, z)` of every pixel (NaN where
  nothing was sampled).

Options:

* `--max-pull F`: Traces whose turn is in doubt (pulled further than F sigma by the labelling) are left out
  (default `3`; `0`: none is).
* `--max-gap F`, `--max-turn F`: A trace carries on from the end of another across a gap of at most F voxels
  (default `60`), turning by at most F degrees (default `45`).
* `--fill-px F`: Holes along the sheet up to F voxels long are bridged straight (default `30`).
* `--fill-z N`: A column with no sheet position in a slice takes the one of the nearest slice, at most N away, that
  knows it (the sheet moves little along Z), instead of staying black (default `0`: never).
* `--memory-slices N`: A trace finds its sheet in the last N slices placed, the newest first, so a region survives a
  trace missing for up to N-1 slices instead of restarting at u = 0 (default `1`: the previous slice only).
* `--inherit-radius F`: A point finds the sheet of an earlier slice within F voxels (default `4`).
* `--min-region-share F`: Regions smaller than this share of the largest one (by area) are left out (default `0.02`).
* `--min-coverage P`: A layer whose sheet position is known for less than P percent of its image (after `--fill-z`)
  is skipped once the sheet has been followed, before the CT volume is read (default `0`: none is).
* `--diagnose-only`: Follows the sheet and prints the report (traces left out by reason, regions, coverage) without
  reading the CT volume or writing any image; `--scroll` is not needed. It also writes `<stem>_coverage.png`, what the
  image would show: white where the slice knows the sheet, grey where `--fill-z` fills in, black where nothing is known.
* `--cache-mb F`: Only the volume chunks the slab actually crosses are read, one layer of the chunk grid along Z at a
  time; at most F MB of decoded chunks are held at once (default `1024`).
* `--slab N` (MIP thickness, default `6`), `--step-z N`, `--max-width N` (default `1600`), `--z-min` / `--z-max`,
  `--threads N`, `--connections N`.

### 3D export (`export_3d`)

```sh
export_3d --scroll-name dataset
export_3d --scroll-name dataset --track 3 --z-min 8000 --z-max 8100
```

Writes a colored point cloud to `results/3D/ply/NAME_pointcloud.ply` and a surface mesh to
`results/3D/obj/scroll_NAME.obj` (both open in Blender, MeshLab or CloudCompare). The mesh joins adjacent rays for
every (layer, summit) pair.

Options:

* `--results-dir DIR`: Folder containing the JSON results (default `results/json/NAME/`).
* `--min-points N`: Minimum number of points of a layer seen from one summit to mesh it (default `60`).
* `--track N`: Keep only the rays of this summit (default: all).
* `--z-min N` / `--z-max N`: Slices kept (default: all).
* `--threads N`: Threads reading the JSON files (`0` = all available).

### Overlay check (`overlay_sections`)

```sh
overlay_sections --scroll-name dataset --coords results/png/dataset/layer_0/unrolled_layer_0_coords.npy
```

Draws the path of an unrolled layer onto a few raw CT slices, to check that it stays on one sheet. Steps longer than
`--jump` times the median step are marked as jumps.

Options:

* `--coords FILE`: The `_coords.npy` file written by `unroll_layer` (required).
* `--scroll URL`: The CT volume given to `unroll_layer` (default: the one recorded for the scroll); a store or any of
  its levels, the level with the surface shape being found by itself.
* `--scroll-name NAME`, `--results-dir DIR`: Where the recorded volumes are found (default `results/json/NAME/`).
* `--sections N`: Slices drawn, evenly spaced over the image (default `8`), or `--z Z1,Z2,...` to pick them.
* `--output DIR`: Output folder (default `sections/` next to the coords file).
* `--margin PX` (default `150`), `--max-size PX` (default `2500`), `--jump F` (default `5`), `--threads N`,
  `--connections N` (default `32`).

`unroll_layer` and `export_3d` read every `<name>_z<z>.json` file of their folder, align the labels along Z, and
support `--z-min` / `--z-max` filtering.

## Architecture & Implementation Notes

A slice is read like a contour map: every sheet is a contour line, and its winding is its height.

* **Summits:** The orientation field of the sheets (structure tensor) is searched for singularities with the Poincaré
  index: index +1 is a core (the heart of a spiral, or a dome), +1/2 the tip of a fold. Summits are tracked across
  slices within `--track-radius` (persisting up to `--track-memory` missing slices). Each summit casts `--rays`
  fixed-angle rays (default `360`), uniquely indexing every ray by `(track, ray_index)` in JSON outputs.
* **Sheet pieces:** The sheets are thinned to a skeleton (Zhang-Suen), cut at junctions into branches, and broken
  gaps are bridged. Branches are split into rigid pieces; probes along the normal of every piece find the next sheet
  on each side (a step of one turn) or the outside of the volume.
* **Orientation, then windings:** Which side of every piece is uphill is solved first (a sign per piece), then the
  winding of every piece (an integer). Around a core, the turn count also changes by one across a seam, in the
  direction the spiral is rolled; every core is tried both ways, as well as no centre at all, and the trial leaving
  the fewest observations unmet is kept.
* **Solver:** Quality-guided voting, as in phase unwrapping: the observations between two pieces are pooled, labels
  grow from the anchors along the most confident links first (maximum spanning tree), then every piece takes the
  label most supported by its neighbours. Disagreements end up on the weakest links instead of being smeared.
* **Continuity along Z:** Every slice hands its solved sheet points to the next one; a point lying within 2 px of one
  of them votes for the same winding, through a free offset so that only the relations between pieces are carried
  over. This settles what a single slice leaves ambiguous.
* **Sheet traces:** Besides the crossings of the rays, every result file holds the solved sheets themselves
  (`"sheets"`): the skeleton walked branch by branch and cut wherever the winding changes, one point every 3 px.
* **Pull:** Every crossing and every trace carries the share of its pieces' observations left unmet, written as
  sigmas; `unroll_layer --max-pull` leaves out the doubtful ones.
* **Unrolling along the sheet:** In every slice, each trace of the layer is rigid along itself (position u = offset
  ± arc length). It inherits u from the sheet of the previous slice where it lies on it, or carries on the u of the
  trace it continues across a junction or a break; the first slice starts from its longest trace.
* **Label alignment:** Labels are aligned between consecutive slices by majority vote over matched points of the same rays; a shift is applied only when enough points agree.
* **Zarr I/O:** Built-in Zarr v2/v3 reader (unsharded, C-order, little-endian; `none`/`blosc`/`zstd`/`gzip`) supporting local paths, `http(s)://`, and anonymous `s3://`. Chunk planes are fetched in parallel and cached across slices.
* **Rendering:** Annotated outputs are drawn directly via OpenCV.