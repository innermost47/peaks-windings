// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>

namespace pw
{
    // Neighbourhood code of a pixel: bit k set when its neighbour k is set, the neighbours being taken clockwise from
    // the top (N, NE, E, SE, S, SW, W, NW). `p` points to the pixel in a CV_8U image of row stride `step`.
    inline int neighbour_code(const unsigned char *p, size_t step)
    {
        const unsigned char *up = p - step, *down = p + step;
        return (up[0] != 0) | (up[1] != 0) << 1 | (p[1] != 0) << 2 | (down[1] != 0) << 3 | (down[0] != 0) << 4 |
               (down[-1] != 0) << 5 | (p[-1] != 0) << 6 | (up[-1] != 0) << 7;
    }

    // Number of separate runs of set neighbours around a pixel (the crossing number): 1 on a line, 2 where a line
    // passes through, 3 or more where lines meet.
    int crossing_number(int code);

    // Zhang-Suen thinning of a binary image (any non-zero pixel is set) to lines one pixel wide. Returns a 0 / 1 CV_8U
    // image of the same size; the outermost row and column of pixels are always 0.
    cv::Mat thin_lines(const cv::Mat &binary, int threads);
} // namespace pw
