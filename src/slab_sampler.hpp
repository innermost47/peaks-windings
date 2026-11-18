// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "rail.hpp"
#include "zarr_reader.hpp"

#include <opencv2/core.hpp>
#include <functional>
#include <vector>

namespace pw
{
    struct SlabParams
    {
        int slab = 6;            // thickness across the sheet of the maximum intensity projection (voxels)
        double cache_mb = 1024;  // memory budget of the decoded chunks held at once
        int threads = 0;         // 0: all hardware threads
    };

    // What was sampled: one row per slice, one column per point of the rails.
    struct SlabImage
    {
        cv::Mat values; // CV_64F, the maximum across the sheet; 0 where nothing was sampled
        cv::Mat coords; // CV_32FC3, the volume coordinates (x, y, z) of every pixel; NaN where nothing was sampled
    };

    // Samples the volume along the rails of the slices `dense_z` (rail(z) gives `width` points), as a maximum
    // intensity projection across the sheet. Only the chunks the slab actually crosses are read: one layer of the chunk
    // grid along Z at a time, its columns in runs whose chunks fit the memory budget (consecutive runs share theirs).
    SlabImage sample_slab(const std::vector<int> &dense_z, int width, const std::function<Rail(int)> &rail,
                          const SlabParams &o, ZarrVolume &volume);
} // namespace pw
