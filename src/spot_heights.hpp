// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pw
{
    // One point of a sheet with its solved winding, as the spot heights of a map.
    struct SpotHeight
    {
        float x = 0, y = 0;
        int winding = 0;
    };

    // The spot heights left by a slice, to be read by the next one: the sheets move little from one slice to the
    // next, so a sheet point of the next slice lying on one of them is most likely on the same turn.
    class SpotHeights
    {
    public:
        SpotHeights() = default;
        // `radius`: full-res px, how far a point may lie from a spot height and still read it.
        explicit SpotHeights(std::vector<SpotHeight> spots, double radius = 2.0);

        bool empty() const { return spots_.empty(); }
        size_t size() const { return spots_.size(); }

        // The winding of the nearest spot height within the radius of (x, y), if any.
        std::optional<int> find(double x, double y) const;

    private:
        std::uint64_t cell_key(int cx, int cy) const
        {
            return static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) << 32 | static_cast<std::uint32_t>(cy);
        }
        int cell_of(double v) const { return static_cast<int>(v / radius_ + (v < 0 ? -1 : 0)); }

        std::vector<SpotHeight> spots_; // sorted by cell
        std::unordered_map<std::uint64_t, std::pair<int, int>> cells_; // cell -> [begin, end) in spots_
        double radius_ = 2.0;
    };
} // namespace pw
