# peaks_windings (C++)

C++ implementation of the layered volume segmentation, unrolling, and 3D meshing pipeline.

| Tool             | Purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `peaks_windings` | Computes consistent winding labels across Z-slices (`<name>_z<z>.json`) |
| `unroll_layer`   | Flattens a single winding layer into a 2D image (slab MIP)              |
| `export_3d`      | Exports layer geometry as PLY point clouds and OBJ meshes               |

## Build

Requires OpenCV (`core`, `imgproc`, `imgcodecs`), Eigen 3, libcurl, c-blosc, and nlohmann-json (`zstd` and `zlib` are optional).

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
sudo apt install build-essential cmake ninja-build libopencv-dev libeigen3-dev libcurl4-openssl-dev \
                 libblosc-dev libzstd-dev zlib1g-dev nlohmann-json3-dev
cmake -S . -B build -G Ninja
cmake --build build
```

Pass `-DPW_NATIVE=ON` to enable `-march=native`.

## Usage

### Label extraction (`peaks_windings`)

```sh
peaks_windings --z-start 8000 --z-count 100 --every 1
peaks_windings --volume-url D:/data/volume.zarr --z-start 0 --z-count 50
peaks_windings --image slice.png
peaks_windings --synthetic

```

Options:

* `--threads N`: Worker threads (`0` = all available).
* `--centre-jobs N`: Concurrent candidate centers evaluated (default `4`; reduce to lower RAM usage).
* `--connections N`: Parallel HTTP downloads (default `32`).
* `--cache-gb F`: Memory budget for decoded chunk planes (default `2` GiB).
* `--image FILE`: Process a single image instead of a Zarr store.
* `--timings`: Log per-step execution times.

### Unrolling and 3D export

```sh
unroll_layer --results-dir results/dataset --volume <zarr_path_or_url> [--layer N]
export_3d    --results-dir results/dataset --obj-out out/surface.obj --ply-out out/points.ply

```

Both tools read `*_z*.json` files, align labels along Z, and support `--z-min` / `--z-max` filtering.

* **`unroll_layer`** outputs `<output>_track<N>_full.png` (8-bit unrolled image), `<output>_track<N>_coords.npy` (float32 `(rows, cols, 3)` volume coordinates for stitching tracks), and `<output>_track<N>.png` (annotated preview). Flags: `--tracks` (default `5`), `--track N`, `--max-width` (default `1600`), `--max-jump` (default `5`x median step).
* **`export_3d`** generates meshes between adjacent rays per tracked summit (`--track N` isolates a single track).

## Architecture & Implementation Notes

* **Summit tracking:** Local maxima (furthest from sheet boundaries) are tracked across slices within `--track-radius` (persisting up to `--track-memory` missing slices). Each summit casts `--rays` fixed-angle rays (default `360`), uniquely indexing every ray by `(track, ray_index)` in JSON outputs.
* **Solver:** Weighted least squares (`lsq`) via Eigen sparse Cholesky decomposition with weak zero-regularization.
* **Zarr I/O:** Built-in Zarr v2/v3 reader (unsharded, C-order, little-endian; `none`/`blosc`/`zstd`/`gzip`) supporting local paths, `http(s)://`, and anonymous `s3://`. Chunk planes are fetched in parallel and cached across slices.
* **Rendering:** Annotated outputs are drawn directly via OpenCV.

## Repository Layout

```
vcpkg.json
LICENSE
README.md
CMakeLists.txt   
results/.gitkeep     
src/main.cpp          CLI entry point, synthetic test, and slice loop
src/windings.cpp/hpp  Core pipeline (masking, spires, peaks, crossings, solver, JSON)
src/unroll_layer.cpp  Layer flattening and coordinate mapping
src/export_3d.cpp     PLY/OBJ mesh generation
src/layers.cpp/hpp    JSON ingestion and Z-axis label alignment
src/zarr_reader.*     Local/remote Zarr volume reader
src/parallel.hpp      Lightweight thread pool
src/turbo.hpp         Turbo colormap LUT
```
