// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <vector>

namespace pw
{
    // The sheet in one slice, one point per column of the unrolled image, with its unit normal (NaN where the sheet
    // is unknown).
    struct Rail
    {
        std::vector<double> xs, ys, nx, ny; // the sheet in one slice and its unit normal, at every flat position
        std::vector<int> zs;
    };

} // namespace pw
