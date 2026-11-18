// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "ray_crossings.hpp"

#include "numeric.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>

namespace pw
{
    namespace
    {
        std::vector<Crossing> ray_crossings(const cv::Mat &spires, const cv::Mat &inside, double x0, double y0,
                                            double angle, double max_radius, int min_spire, int min_gap)
        {
            const int h = spires.rows, w = spires.cols;
            const double ca = std::cos(angle), sa = std::sin(angle);
            std::vector<unsigned char> s;
            std::vector<double> xs, ys;
            for (int i = 0; 1.0 + i < max_radius; ++i)
            {
                const double r = 1.0 + i;
                const double x = x0 + r * ca, y = y0 + r * sa;
                const int ix = round_half_even(x), iy = round_half_even(y);
                if (ix < 0 || ix >= w || iy < 0 || iy >= h || !inside.at<unsigned char>(iy, ix))
                    break;
                s.push_back(spires.at<unsigned char>(iy, ix));
                xs.push_back(x);
                ys.push_back(y);
            }
            std::vector<Crossing> out;
            const int n = static_cast<int>(s.size());
            int gap = min_gap; // the start of the ray counts as void
            for (int k = 0; k < n;)
            {
                int j = k;
                if (s[k])
                {
                    while (j < n && s[j])
                        ++j;
                    if (j - k >= min_spire && gap >= min_gap)
                    {
                        const int m = (k + j - 1) / 2;
                        out.push_back({xs[m], ys[m], 1.0 + m});
                        gap = 0;
                    }
                    // else: a sheet straight after a too-short void is the same sheet
                }
                else
                {
                    while (j < n && !s[j])
                        ++j;
                    gap = j - k;
                }
                k = j;
            }
            return out;
        }
    } // namespace

    std::vector<Ray> shoot_rays(const cv::Mat &spires, const cv::Mat &inside, const std::vector<Peak> &peaks,
                                const RayParams &params, int threads)
    {
        const int n = std::max(params.n_rays, 1);
        const double max_radius = params.max_radius > 0 ? params.max_radius : std::max(spires.rows, spires.cols);
        const double step = TWO_PI / n;
        std::vector<Ray> rays;
        rays.reserve(peaks.size() * n);
        for (int p = 0; p < static_cast<int>(peaks.size()); ++p)
            for (int i = 0; i < n; ++i)
                rays.push_back({p, i, i * step, {}});
        parallel_for(static_cast<int>(rays.size()), threads, [&](int i, int)
                     {
                         const Peak &p = peaks[rays[i].peak];
                         rays[i].crossings = ray_crossings(spires, inside, p.x, p.y, rays[i].angle, max_radius,
                                                           params.min_spire, params.min_gap); });
        return rays;
    }
} // namespace pw
