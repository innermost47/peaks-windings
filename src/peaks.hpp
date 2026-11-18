// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <vector>

namespace pw
{
    struct Peak
    {
        double x = 0, y = 0, height = 0;
        int track = 0; // identity of the summit along z (its rank in the slice when summits are not tracked)
    };

    // The summits followed from slice to slice. A summit keeps its track number as long as it is found again close to
    // where it was; a summit that appears gets a new number.
    struct TrackState
    {
        struct Track
        {
            int id = 0;
            double x = 0, y = 0;
            int missed = 0;
        };
        std::vector<Track> tracks;
        int next_id = 0;
    };

    // The n summits of the slice, chosen among the candidates (highest first) and given their track numbers. A
    // candidate close to a tracked summit is that summit and keeps its place; the free places go to the highest other
    // candidates, which start new tracks. Without a state, the n highest are taken and numbered by height.
    std::vector<Peak> select_peaks(const std::vector<Peak> &candidates, int n, TrackState *state, double radius,
                                   int memory);
} // namespace pw
