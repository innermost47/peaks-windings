// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "point_grid.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace pw
{
    // One point of a sheet with its solved winding (`value`), as the spot heights of a map.
    using SpotHeight = PointGrid<int>::Entry;

    // The spot heights left by a slice, to be read by the next one: the sheets move little from one slice to the
    // next, so a sheet point of the next slice lying on one of them is most likely on the same turn.
    class SpotHeights
    {
    public:
        SpotHeights() = default;
        // `radius`: full-res px, how far a point may lie from a spot height and still read it.
        explicit SpotHeights(std::vector<SpotHeight> spots, double radius = 2.0) : grid_(std::move(spots), radius) {}

        bool empty() const { return grid_.empty(); }
        size_t size() const { return grid_.size(); }

        // The winding of the nearest spot height within the radius of (x, y), if any.
        std::optional<int> find(double x, double y) const
        {
            const SpotHeight *spot = grid_.nearest(x, y);
            return spot ? std::optional<int>(spot->value) : std::nullopt;
        }

    private:
        PointGrid<int> grid_;
    };
} // namespace pw
