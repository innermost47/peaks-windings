// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "sheet_skeleton.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace pw
{
    // A stretch of one sheet on one turn: its centre line, point after point along the sheet.
    struct SheetTrace
    {
        int winding = 0;
        float pull = 0; // the largest pull (sigmas) of the pieces it runs over
        std::vector<cv::Point> points;
    };

    struct TraceParams
    {
        int min_length_px = 20; // shorter stretches are left out
        int every_px = 3;       // one node of the skeleton in this many is kept (and the last one)
    };

    // The sheets of a slice as traces: every branch of the skeleton is walked from its far end (a branch closed on
    // itself, or with spurs, gives one trace per arm), and cut wherever its winding changes or is unknown.
    // `winding(node)` is the solved winding of a node (none where it was not solved), `pull(node)` its pull.
    std::vector<SheetTrace> trace_sheets(const SheetSkeleton &skeleton,
                                         const std::function<std::optional<int>(int)> &winding,
                                         const std::function<float(int)> &pull, const TraceParams &params = {});
} // namespace pw
