// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "peaks.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace pw
{
    // Where a ray goes through a sheet: the middle of the run of spire pixels; r is the distance from the summit.
    struct Crossing
    {
        double x, y, r;
    };

    struct Ray
    {
        int peak;  // index of the summit in the slice
        int index; // index of the ray around its summit: it always leaves at the angle index * 2 pi / n
        double angle;
        std::vector<Crossing> crossings; // in order along the ray
    };

    struct RayParams
    {
        int n_rays = 360;
        double max_radius = 0; // full-res px; 0: the size of the slice
        int min_spire = 2;     // px of spire for a crossing
        int min_gap = 3;       // px of void before a new crossing
    };

    // The same n rays from every summit, each followed until it leaves the interior.
    std::vector<Ray> shoot_rays(const cv::Mat &spires, const cv::Mat &inside, const std::vector<Peak> &peaks,
                                const RayParams &params, int threads);
} // namespace pw
