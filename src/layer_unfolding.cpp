// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "layer_unfolding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <limits>

namespace pw
{
    namespace
    {
        const double NaN = std::numeric_limits<double>::quiet_NaN();
        const double DEG = 3.14159265358979323846 / 180.0;

        double median(std::vector<double> &v)
        {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        }
    } // namespace

    LayerUnfolding::Polyline LayerUnfolding::polyline(const Trace &trace)
    {
        Polyline line;
        const size_t n = trace.size();
        line.x.resize(n);
        line.y.resize(n);
        line.s.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            line.x[i] = trace.xy[2 * i];
            line.y[i] = trace.xy[2 * i + 1];
            line.s[i] = i == 0 ? 0.0 : line.s[i - 1] + std::hypot(line.x[i] - line.x[i - 1], line.y[i] - line.y[i - 1]);
        }
        return line;
    }

    LayerUnfolding::Placed LayerUnfolding::place(const Polyline &line, int region, double offset, int direction)
    {
        Placed p;
        static_cast<Polyline &>(p) = line;
        p.region = region;
        p.u.resize(line.s.size());
        for (size_t i = 0; i < line.s.size(); ++i)
            p.u[i] = static_cast<float>(offset + direction * line.s[i]);
        return p;
    }

    LayerUnfolding::End LayerUnfolding::end_of(const Polyline &line, bool last) const
    {
        const size_t n = line.s.size();
        const size_t tip = last ? n - 1 : 0;
        // the point end_tangent_px inside the trace
        size_t inner = tip;
        if (last)
            while (inner > 0 && line.s[tip] - line.s[inner] < params_.end_tangent_px)
                --inner;
        else
            while (inner + 1 < n && line.s[inner] - line.s[tip] < params_.end_tangent_px)
                ++inner;
        End e;
        e.x = line.x[tip];
        e.y = line.y[tip];
        const double dx = line.x[tip] - line.x[inner], dy = line.y[tip] - line.y[inner];
        const double norm = std::hypot(dx, dy);
        if (norm > 0)
        {
            e.dx = dx / norm;
            e.dy = dy / norm;
        }
        return e;
    }

    const LayerUnfolding::Spot *LayerUnfolding::recall(double x, double y) const
    {
        for (const PointGrid<Spot> &grid : memory_)
            if (const auto *entry = grid.nearest(x, y))
                return &entry->value;
        return nullptr;
    }

    // Traces lying on the sheet of the previous slice take the region and the u they find there, as a rigid fit.
    void LayerUnfolding::inherit(const std::vector<Polyline> &candidates, std::vector<char> &done,
                                 std::vector<Placed> &placed)
    {
        if (memory_.empty())
            return;
        struct Fit
        {
            size_t index;
            int region;
            double offset;
            int direction;
            size_t support;
        };
        std::vector<Fit> fits;
        std::vector<double> s, u, residual;
        std::vector<std::pair<size_t, Spot>> found;
        std::unordered_map<int, size_t> votes;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const Polyline &c = candidates[i];
            found.clear();
            votes.clear();
            for (size_t k = 0; k < c.s.size(); ++k)
                if (const Spot *spot = recall(c.x[k], c.y[k]))
                {
                    found.emplace_back(k, *spot);
                    ++votes[spot->region];
                }
            if (found.empty())
                continue;
            // the region most of its points find
            const auto top = std::max_element(votes.begin(), votes.end(), [](const auto &a, const auto &b)
                                              { return a.second < b.second; });
            const int region = top->first;
            s.clear();
            u.clear();
            for (const auto &[k, spot] : found)
                if (spot.region == region)
                {
                    s.push_back(c.s[k]);
                    u.push_back(spot.u);
                }
            if (s.size() < 3 || s.size() < params_.min_inherited * c.s.size())
                continue;
            Fit best{i, region, 0, 0, s.size()};
            double best_misfit = std::numeric_limits<double>::infinity();
            for (int direction : {1, -1})
            {
                residual.resize(s.size());
                for (size_t k = 0; k < s.size(); ++k)
                    residual[k] = u[k] - direction * s[k];
                const double offset = median(residual);
                for (size_t k = 0; k < s.size(); ++k)
                    residual[k] = std::abs(u[k] - direction * s[k] - offset);
                const double misfit = median(residual);
                if (misfit < best_misfit)
                {
                    best_misfit = misfit;
                    best.offset = offset;
                    best.direction = direction;
                }
            }
            if (best_misfit <= params_.max_misfit_px)
                fits.push_back(best);
        }
        // the best supported first: they prevail where traces overlap
        std::stable_sort(fits.begin(), fits.end(), [](const Fit &a, const Fit &b)
                         { return a.support > b.support; });
        for (const Fit &f : fits)
        {
            placed.push_back(place(candidates[f.index], f.region, f.offset, f.direction));
            done[f.index] = 1;
            ++stats_.inherited;
        }
    }

    // Free ends of placed traces carried on by the unplaced traces that go on from them, the closest and straightest
    // first, until no end can be carried on.
    void LayerUnfolding::join(const std::vector<Polyline> &candidates, std::vector<char> &done,
                              std::vector<Placed> &placed)
    {
        const double straight = std::cos(params_.max_turn_deg * DEG);
        std::vector<std::array<char, 2>> used(placed.size(), {0, 0});
        while (true)
        {
            double best_cost = std::numeric_limits<double>::infinity();
            size_t best_i = 0, best_p = 0;
            bool best_last_i = false, best_last_p = false;
            double best_gap = 0;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (done[i])
                    continue;
                for (bool last_i : {false, true})
                {
                    const End ei = end_of(candidates[i], last_i);
                    for (size_t p = 0; p < placed.size(); ++p)
                        for (bool last_p : {false, true})
                        {
                            if (used[p][last_p])
                                continue;
                            const End ep = end_of(placed[p], last_p);
                            const double gx = ei.x - ep.x, gy = ei.y - ep.y, gap = std::hypot(gx, gy);
                            if (gap > params_.max_gap_px)
                                continue;
                            // the trace goes on from the end: it comes in (against its own way out) the way p leaves
                            const double turn = -(ep.dx * ei.dx + ep.dy * ei.dy);
                            if (turn < straight)
                                continue;
                            // and lies ahead of it (beyond the few px where two sheets meet at a junction)
                            if (gap > 10 && (gx * ep.dx + gy * ep.dy) / gap < straight)
                                continue;
                            const double cost = gap + params_.max_gap_px * (1 - turn);
                            if (cost < best_cost)
                            {
                                best_cost = cost;
                                best_i = i;
                                best_p = p;
                                best_last_i = last_i;
                                best_last_p = last_p;
                                best_gap = gap;
                            }
                        }
                }
            }
            if (!std::isfinite(best_cost))
                return;
            const Placed &from = placed[best_p];
            const size_t tip = best_last_p ? from.u.size() - 1 : 0;
            const size_t inner = best_last_p ? (tip > 0 ? tip - 1 : 0) : std::min<size_t>(1, from.u.size() - 1);
            // u keeps the sense it has towards that end of p
            const int sense = from.u[tip] >= from.u[inner] ? 1 : -1;
            const Polyline &line = candidates[best_i];
            const double start = from.u[tip] + sense * best_gap;
            const int region = from.region;
            placed.push_back(best_last_i ? place(line, region, start + sense * line.length(), -sense)
                                         : place(line, region, start, sense));
            used[best_p][best_last_p] = 1;
            used.push_back({0, 0});
            used.back()[best_last_i] = 1;
            done[best_i] = 1;
            ++stats_.joined;
        }
    }

    // The longest traces left start regions of their own, each carried on by the traces that go on from it.
    void LayerUnfolding::seed(const std::vector<Polyline> &candidates, std::vector<char> &done,
                              std::vector<Placed> &placed)
    {
        while (true)
        {
            int longest = -1;
            for (size_t i = 0; i < candidates.size(); ++i)
                if (!done[i] && candidates[i].length() >= params_.min_seed_px &&
                    (longest < 0 || candidates[i].length() > candidates[longest].length()))
                    longest = static_cast<int>(i);
            if (longest < 0)
                return;
            const int region = static_cast<int>(regions_.size());
            regions_.emplace_back();
            placed.push_back(place(candidates[longest], region, 0.0, 1));
            done[longest] = 1;
            ++stats_.seeded;
            join(candidates, done, placed);
        }
    }

    void LayerUnfolding::print_region_seams() const
    {
        std::printf("--> Coexistence of regions (minimum distance between traces at different Z levels):\n");
        for (int z : {4020, 4180, 4340, 4500})
        {
            const auto it = slices_.find(z);
            if (it == slices_.end())
                continue;
            for (size_t a = 0; a < regions_.size(); ++a)
                for (size_t b = a + 1; b < regions_.size(); ++b)
                {
                    bool pa_found = false, pb_found = false;
                    double best = 1e9;
                    for (const Placed &p1 : it->second)
                    {
                        if (p1.region != (int)a)
                            continue;
                        pa_found = true;
                        for (const Placed &p2 : it->second)
                        {
                            if (p2.region != (int)b)
                                continue;
                            pb_found = true;
                            for (size_t i = 0; i < p1.x.size(); ++i)
                                for (size_t j = 0; j < p2.x.size(); ++j)
                                    best = std::min(best, std::hypot<double>(p1.x[i] - p2.x[j], p1.y[i] - p2.y[j]));
                        }
                    }
                    if (pa_found && pb_found && best < 200)
                        std::printf("   - z=%d R%zu-R%zu: gap = %.1f voxels\n", z, a, b, best);
                }
        }
    }

    void LayerUnfolding::add_slice(int z, const std::vector<Trace> &traces)
    {
        std::vector<Polyline> candidates;
        stats_.offered += static_cast<long>(traces.size());
        for (const Trace &t : traces)
        {
            if (params_.max_pull > 0 && t.pull > params_.max_pull)
            {
                ++stats_.dropped;
                ++stats_.pulled;
                continue;
            }
            Polyline line = polyline(t);
            if (line.s.size() < 2 || line.length() < params_.min_trace_px)
            {
                ++stats_.dropped;
                ++stats_.too_short;
                continue;
            }
            candidates.push_back(std::move(line));
        }
        if (candidates.empty())
            return;

        std::vector<char> done(candidates.size(), 0);
        std::vector<Placed> placed;
        inherit(candidates, done, placed);
        join(candidates, done, placed);
        seed(candidates, done, placed);
        const long left_over = static_cast<long>(std::count(done.begin(), done.end(), 0));
        stats_.dropped += left_over;
        stats_.unplaced += left_over;
        if (placed.empty())
            return;
        ++stats_.slices;

        std::vector<PointGrid<Spot>::Entry> entries;
        for (const Placed &p : placed)
        {
            Region &r = regions_[p.region];
            r.area += p.length();
            r.z_first = std::min(r.z_first, z);
            r.z_last = std::max(r.z_last, z);
            for (size_t k = 0; k < p.u.size(); ++k)
            {
                entries.push_back({p.x[k], p.y[k], Spot{p.region, p.u[k]}});
                x_max_ = std::max<double>(x_max_, p.x[k]);
                y_max_ = std::max<double>(y_max_, p.y[k]);
                r.u_min = std::min<double>(r.u_min, p.u[k]);
                r.u_max = std::max<double>(r.u_max, p.u[k]);
            }
        }
        memory_.emplace_front(std::move(entries), params_.inherit_radius_px);
        while (static_cast<int>(memory_.size()) > std::max(1, params_.memory_slices))
            memory_.pop_back();
        slices_[z] = std::move(placed);
    }

    void LayerUnfolding::lay_out()
    {
        std::vector<int> order(regions_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b)
                         { return regions_[a].area > regions_[b].area; });
        const double largest = order.empty() ? 0.0 : regions_[order.front()].area;
        double next = 0;
        stats_.regions_kept = 0;
        stats_.regions_total = static_cast<long>(regions_.size());
        for (int r : order)
        {
            Region &region = regions_[r];
            if (region.area < params_.min_region_share * largest || !(region.u_max > region.u_min))
                continue;
            region.offset = next;
            next += region.u_max - region.u_min + params_.region_gap_px;
            ++stats_.regions_kept;
        }
        width_ = std::max(0.0, next - params_.region_gap_px);
    }

    Rail LayerUnfolding::rail(int z, const std::vector<double> &u, bool fill_z) const
    {
        Rail r = slice_rail(z, u);
        if (!fill_z || params_.fill_z < 1)
            return r;
        // columns unknown in this slice take the sheet of the nearest slice that knows them (the columns of a region
        // are the same in every slice)
        size_t missing = 0;
        for (double x : r.xs)
            missing += !std::isfinite(x);
        for (int d = 1; d <= params_.fill_z && missing > 0; ++d)
            for (int side : {-1, 1})
            {
                if (slices_.find(z + side * d) == slices_.end())
                    continue;
                const Rail near = slice_rail(z + side * d, u);
                for (size_t c = 0; c < u.size(); ++c)
                    if (!std::isfinite(r.xs[c]) && std::isfinite(near.xs[c]))
                    {
                        r.xs[c] = near.xs[c];
                        r.ys[c] = near.ys[c];
                        r.nx[c] = near.nx[c];
                        r.ny[c] = near.ny[c];
                        r.zs[c] = near.zs[c];
                        --missing;
                    }
            }
        return r;
    }

    Rail LayerUnfolding::slice_rail(int z, const std::vector<double> &u) const
    {
        const size_t n = u.size();
        Rail r;
        r.xs.assign(n, NaN);
        r.ys.assign(n, NaN);
        r.nx.assign(n, NaN);
        r.ny.assign(n, NaN);
        r.zs.assign(n, std::numeric_limits<int>::min());
        const auto it = slices_.find(z);
        if (it == slices_.end() || n < 2)
            return r;
        std::vector<int> region_of(n, -1);
        const double u0 = u.front(), du = (u.back() - u.front()) / (n - 1);
        auto set = [&](size_t c, double x, double y, double tx, double ty)
        {
            const double norm = std::hypot(tx, ty);
            if (norm <= 0)
                return;
            r.xs[c] = x;
            r.ys[c] = y;
            r.nx[c] = -ty / norm;
            r.ny[c] = tx / norm;
            r.zs[c] = z;
        };
        for (const Placed &p : it->second)
        {
            const Region &region = regions_[p.region];
            if (region.offset < 0)
                continue;
            const double shift = region.offset - region.u_min; // from the u of the region to the image
            for (size_t k = 0; k + 1 < p.u.size(); ++k)
            {
                const double ua = p.u[k] + shift, ub = p.u[k + 1] + shift;
                if (ua == ub)
                    continue;
                const double lo = std::min(ua, ub), hi = std::max(ua, ub);
                const long c0 = std::max(0L, static_cast<long>(std::ceil((lo - u0) / du)));
                const long c1 = std::min(static_cast<long>(n) - 1, static_cast<long>(std::floor((hi - u0) / du)));
                // the tangent in the sense of u
                const double sense = ub > ua ? 1.0 : -1.0;
                const double tx = sense * (p.x[k + 1] - p.x[k]), ty = sense * (p.y[k + 1] - p.y[k]);
                for (long c = c0; c <= c1; ++c)
                {
                    if (std::isfinite(r.xs[c]))
                        continue; // a trace placed before prevails
                    const double t = (u[c] - ua) / (ub - ua);
                    set(c, p.x[k] + t * (p.x[k + 1] - p.x[k]), p.y[k] + t * (p.y[k + 1] - p.y[k]), tx, ty);
                    region_of[c] = p.region;
                }
            }
        }
        // short holes along the sheet of one region: straight across
        const long max_hole = static_cast<long>(params_.fill_px / du);
        long left = -1;
        for (long c = 0; c < static_cast<long>(n); ++c)
        {
            if (region_of[c] < 0)
                continue;
            if (left >= 0 && c - left > 1 && c - left - 1 <= max_hole && region_of[left] == region_of[c])
            {
                const double tx = r.xs[c] - r.xs[left], ty = r.ys[c] - r.ys[left];
                for (long k = left + 1; k < c; ++k)
                {
                    const double t = static_cast<double>(k - left) / (c - left);
                    set(k, r.xs[left] + t * tx, r.ys[left] + t * ty, tx, ty);
                }
            }
            left = c;
        }
        return r;
    }

    std::vector<LayerUnfolding::RegionSummary> LayerUnfolding::largest_regions(size_t top) const
    {
        std::vector<int> order(regions_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b)
                         { return regions_[a].area > regions_[b].area; });
        std::vector<RegionSummary> out;
        for (size_t i = 0; i < order.size() && i < top; ++i)
        {
            const Region &r = regions_[order[i]];
            out.push_back({r.u_max > r.u_min ? r.u_max - r.u_min : 0.0, r.area, r.z_first, r.z_last, r.offset >= 0});
        }
        return out;
    }

    double LayerUnfolding::total_area() const
    {
        double sum = 0;
        for (const Region &r : regions_)
            sum += r.area;
        return sum;
    }

    double LayerUnfolding::kept_area() const
    {
        double sum = 0;
        for (const Region &r : regions_)
            if (r.offset >= 0)
                sum += r.area;
        return sum;
    }

    double LayerUnfolding::coverage(const std::vector<double> &u, bool fill_z) const
    {
        if (slices_.empty() || u.empty())
            return 0.0;
        double total = 0;
        for (const auto &slice : slices_)
        {
            const Rail r = rail(slice.first, u, fill_z);
            total += static_cast<double>(std::count_if(r.xs.begin(), r.xs.end(), [](double x)
                                                       { return std::isfinite(x); })) /
                     static_cast<double>(u.size());
        }
        return total / static_cast<double>(slices_.size());
    }
} // namespace pw
