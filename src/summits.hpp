// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "peaks.hpp"
#include "sheet_directions.hpp"

#include <vector>

namespace pw
{
    // The summits of a slice, read as a contour map: the points the sheets wind around.
    //
    // Going once around a point, the direction of the sheets turns by a whole number of half turns (the Poincare
    // index):
    //   +1    around the core of nested sheets: the heart of the scroll, or a dome of a fold cut across by the slice;
    //   +1/2  at the tip of a fold (sheets bent into a U);
    //   0     anywhere else, including inside a void between sheets that merely run side by side.
    // A core is often elongated, and then shows as two +1/2 close together: cores are therefore read on a wide loop
    // (core_loop_radius_px), the tips of folds on a narrow one (fold_loop_radius_px).
    struct SummitParams
    {
        double fold_loop_radius_px = 90;  // full-res px
        double core_loop_radius_px = 250; // full-res px
        double grid_step_px = 15;         // full-res px between the points the index is measured at
    };

    enum class SummitKind
    {
        core, // index +1
        fold  // index +1/2
    };

    // The summits of a slice, the cores first, each kind the most clearly read first (on the most grid points), in
    // full-resolution pixels. Peak::height holds the index.
    std::vector<Peak> find_summits(const SheetDirections &directions, const SummitParams &params = {});

    inline SummitKind kind_of(const Peak &summit) { return summit.height > 0.75 ? SummitKind::core : SummitKind::fold; }
} // namespace pw
