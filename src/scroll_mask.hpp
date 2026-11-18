// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>

namespace pw
{
    struct Params;

    // The sheets ("spires") of one slice and the interior of the scroll they lie in.
    struct ScrollMasks
    {
        cv::Mat mask_small;   // interior of the scroll, at the working scale (0 / 255)
        cv::Mat spires_small; // spires without the frame, at the working scale (0 / 255)
        cv::Mat spires;       // spires at full resolution, specks removed (0 / 1)
        cv::Mat inside;       // interior at full resolution (0 / 1)
    };

    // Otsu threshold, frame (case) removal and interior detection at the working scale, then the spires at full
    // resolution. Anything thicker than params.frame_px full-resolution pixels is the frame.
    ScrollMasks detect_scroll_masks(const cv::Mat &img_u8, const Params &params);
} // namespace pw
