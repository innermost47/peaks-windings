// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace pw
{
    // An unrolled sheet: one row per slice, one column per position along the sheet.
    struct UnrolledImage
    {
        cv::Mat values;   // CV_64F, 0 where nothing was sampled
        cv::Mat coords;   // CV_32FC3, the volume coordinates (x, y, z) of every pixel; NaN where nothing was sampled
        int z_start = 0, z_end = 0;
        std::string title, x_label;
        double x_max = 0; // what the right edge of the image stands for (x_label's unit)
    };

    // Writes the image contrast-stretched to 8 bits as <stem>_full.png, a figure with graduated axes as `output_png`,
    // and the coordinates of its pixels as <stem>_coords.npy (float32, rows x cols x 3).
    void save_unrolled(const UnrolledImage &im, const std::string &output_png);
} // namespace pw
