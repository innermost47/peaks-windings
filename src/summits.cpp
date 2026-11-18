// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "summits.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace pw
{
    namespace
    {
        struct Reading
        {
            int gx, gy;         // grid cell
            double x, y, index; // full-res px
        };

        struct Summit
        {
            double x = 0, y = 0, index = 0;
            int readings = 0;
        };

        // Every grid point whose loop gives an index in [lo, hi).
        std::vector<Reading> read(const SheetDirections &directions, double radius, int step, double scale, double lo,
                                  double hi)
        {
            const std::vector<cv::Point> loop = SheetDirections::circle(radius);
            std::vector<Reading> readings;
            for (int y = 0; y < directions.size().height; y += step)
                for (int x = 0; x < directions.size().width; x += step)
                {
                    double index = 0;
                    if (directions.index(x, y, loop, index) && index >= lo && index < hi)
                        readings.push_back({x / step, y / step, x / scale, y / scale, index});
                }
            return readings;
        }

        // The readings around one summit are many: every grid point whose loop encloses it, a patch of neighbouring
        // cells. One summit per patch, at its centroid; the most read first.
        std::vector<Summit> gather(const std::vector<Reading> &readings)
        {
            std::map<std::pair<int, int>, int> cell;
            for (int i = 0; i < static_cast<int>(readings.size()); ++i)
                cell[{readings[i].gx, readings[i].gy}] = i;
            std::vector<int> patch(readings.size(), -1);
            std::vector<Summit> summits;
            for (int i = 0; i < static_cast<int>(readings.size()); ++i)
            {
                if (patch[i] >= 0)
                    continue;
                Summit s;
                std::vector<int> stack{i};
                patch[i] = static_cast<int>(summits.size());
                while (!stack.empty())
                {
                    const Reading &r = readings[stack.back()];
                    stack.pop_back();
                    s.x += r.x, s.y += r.y, s.index += r.index, ++s.readings;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            const auto it = cell.find({r.gx + dx, r.gy + dy});
                            if (it != cell.end() && patch[it->second] < 0)
                            {
                                patch[it->second] = patch[i];
                                stack.push_back(it->second);
                            }
                        }
                }
                s.x /= s.readings, s.y /= s.readings, s.index /= s.readings;
                summits.push_back(s);
            }
            std::stable_sort(summits.begin(), summits.end(), [](const Summit &a, const Summit &b)
                             { return a.readings > b.readings; });
            return summits;
        }
    } // namespace

    std::vector<Peak> find_summits(const SheetDirections &directions, const SummitParams &params)
    {
        const double scale = directions.scale();
        const int step = std::max(1, static_cast<int>(std::lround(params.grid_step_px * scale)));
        const std::vector<Summit> cores =
            gather(read(directions, params.core_loop_radius_px * scale, step, scale, 0.75, 1.25));
        const std::vector<Summit> folds =
            gather(read(directions, params.fold_loop_radius_px * scale, step, scale, 0.25, 0.75));

        std::vector<Peak> summits;
        for (const Summit &s : cores)
            summits.push_back({s.x, s.y, s.index, 0});
        // a core read on the wide loop is often two +1/2 on the narrow one: those are the core, not folds
        for (const Summit &s : folds)
        {
            const bool in_core = std::any_of(cores.begin(), cores.end(), [&](const Summit &c)
                                             { return std::hypot(c.x - s.x, c.y - s.y) < params.core_loop_radius_px; });
            if (!in_core)
                summits.push_back({s.x, s.y, s.index, 0});
        }
        return summits;
    }
} // namespace pw
