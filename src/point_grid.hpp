// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pw
{
    // Points of a slice carrying a value, to be looked up by position: the nearest one within a radius.
    template <class Value>
    class PointGrid
    {
    public:
        struct Entry
        {
            float x = 0, y = 0;
            Value value{};
        };

        PointGrid() = default;
        // `radius`: px, how far a position may lie from a point and still find it.
        PointGrid(std::vector<Entry> entries, double radius)
            : entries_(std::move(entries)), radius_(std::max(radius, 0.5))
        {
            auto key_of = [&](const Entry &e)
            { return key(cell(e.x), cell(e.y)); };
            std::sort(entries_.begin(), entries_.end(), [&](const Entry &a, const Entry &b)
                      { return key_of(a) < key_of(b); });
            cells_.reserve(entries_.size() / 4 + 1);
            for (int i = 0; i < static_cast<int>(entries_.size());)
            {
                const std::uint64_t k = key_of(entries_[i]);
                int j = i + 1;
                while (j < static_cast<int>(entries_.size()) && key_of(entries_[j]) == k)
                    ++j;
                cells_.emplace(k, std::make_pair(i, j));
                i = j;
            }
        }

        bool empty() const { return entries_.empty(); }
        size_t size() const { return entries_.size(); }
        double radius() const { return radius_; }

        // The nearest point within the radius of (x, y), nullptr for none.
        const Entry *nearest(double x, double y) const
        {
            const int cx = cell(x), cy = cell(y);
            double best = radius_ * radius_;
            const Entry *found = nullptr;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                {
                    const auto it = cells_.find(key(cx + dx, cy + dy));
                    if (it == cells_.end())
                        continue;
                    for (int i = it->second.first; i < it->second.second; ++i)
                    {
                        const double ex = entries_[i].x - x, ey = entries_[i].y - y;
                        const double d = ex * ex + ey * ey;
                        if (d <= best)
                        {
                            best = d;
                            found = &entries_[i];
                        }
                    }
                }
            return found;
        }

    private:
        static std::uint64_t key(int cx, int cy)
        {
            return static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) << 32 | static_cast<std::uint32_t>(cy);
        }
        int cell(double v) const { return static_cast<int>(std::floor(v / radius_)); }

        std::vector<Entry> entries_; // sorted by cell
        std::unordered_map<std::uint64_t, std::pair<int, int>> cells_; // cell -> [begin, end) in entries_
        double radius_ = 1.0;
    };
} // namespace pw
