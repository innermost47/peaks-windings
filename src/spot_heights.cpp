// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "spot_heights.hpp"

#include <algorithm>

namespace pw
{
    SpotHeights::SpotHeights(std::vector<SpotHeight> spots, double radius)
        : spots_(std::move(spots)), radius_(std::max(radius, 0.5))
    {
        auto key_of = [&](const SpotHeight &s)
        { return cell_key(cell_of(s.x), cell_of(s.y)); };
        std::sort(spots_.begin(), spots_.end(), [&](const SpotHeight &a, const SpotHeight &b)
                  { return key_of(a) < key_of(b); });
        cells_.reserve(spots_.size() / 4 + 1);
        for (int i = 0; i < static_cast<int>(spots_.size());)
        {
            const std::uint64_t key = key_of(spots_[i]);
            int j = i + 1;
            while (j < static_cast<int>(spots_.size()) && key_of(spots_[j]) == key)
                ++j;
            cells_.emplace(key, std::make_pair(i, j));
            i = j;
        }
    }

    std::optional<int> SpotHeights::find(double x, double y) const
    {
        const int cx = cell_of(x), cy = cell_of(y);
        double best = radius_ * radius_;
        std::optional<int> winding;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
            {
                const auto it = cells_.find(cell_key(cx + dx, cy + dy));
                if (it == cells_.end())
                    continue;
                for (int i = it->second.first; i < it->second.second; ++i)
                {
                    const double ex = spots_[i].x - x, ey = spots_[i].y - y;
                    const double d = ex * ex + ey * ey;
                    if (d <= best)
                    {
                        best = d;
                        winding = spots_[i].winding;
                    }
                }
            }
        return winding;
    }
} // namespace pw
