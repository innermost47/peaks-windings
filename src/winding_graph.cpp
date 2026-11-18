// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "winding_graph.hpp"

#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace pw
{
    WindingGraph::WindingGraph(const SheetPieces &pieces, const std::vector<int> &orientation,
                               const SheetSkeleton &skeleton, const std::vector<Ray> &rays,
                               const std::vector<Peak> &peaks, int centre, int handedness, const SpotHeights *spots,
                               const Params &params)
        : pieces_(pieces), params_(params), handedness_(handedness)
    {
        measure_angles(skeleton, peaks, centre);
        add_steps(orientation);
        add_links(skeleton, peaks, centre);
        if (spots && !spots->empty())
            add_spots(skeleton, *spots);
        add_anchors(rays, centre);
    }

    int WindingGraph::best_outermost() const
    {
        // components of the unknowns tied by the observations, weighed by them
        std::vector<int> root(unknowns());
        std::iota(root.begin(), root.end(), 0);
        auto find = [&](int x)
        {
            while (root[x] != x)
                x = root[x] = root[root[x]];
            return x;
        };
        for (const Observation &o : observations_)
            if (o.b >= 0)
                root[find(o.a)] = find(o.b);
        std::vector<double> weight(unknowns(), 0.0);
        for (const Observation &o : observations_)
            weight[find(o.a)] += o.weight;
        int best = -1;
        for (const SheetPieces::Outermost &o : pieces_.outermost())
            if (best < 0 || weight[find(pieces_.piece_of(o.site))] > weight[find(pieces_.piece_of(best))])
                best = o.site;
        return best;
    }

    void WindingGraph::measure_angles(const SheetSkeleton &skeleton, const std::vector<Peak> &peaks, int centre)
    {
        const int n = pieces_.sites();
        phase_.assign(n, 0.0);
        continuous_shift_.assign(n, 0.0);
        shift_.assign(n, 0);
        if (centre < 0)
            return;
        const double cx = peaks[centre].x, cy = peaks[centre].y;
        const std::vector<double> theta = skeleton.unwrapped_angles(cx, cy);
        for (int s = 0; s < n; ++s)
        {
            phase_[s] = angle_around(pieces_.position(s).x, pieces_.position(s).y, cx, cy) / TWO_PI;
            const int node = pieces_.node_of(s);
            if (node < 0)
                continue;
            continuous_shift_[s] = handedness_ * (theta[node] / TWO_PI - phase_[s]);
            shift_[s] = static_cast<int>(std::lround(continuous_shift_[s]));
        }
    }

    void WindingGraph::observe(int a, int b, double dn, double weight)
    {
        const int d = static_cast<int>(std::lround(dn));
        if (std::abs(dn - d) > params_.max_mismatch)
            return;
        const int pa = pieces_.piece_of(a);
        if (b < 0)
        {
            observations_.push_back({pa, -1, d - shift_[a], weight});
            sites_of_.emplace_back(a, -1);
            return;
        }
        const int pb = pieces_.piece_of(b);
        if (pa == pb)
            return; // within one rigid piece: nothing to solve
        observations_.push_back({pa, pb, d - shift_[b] + shift_[a], weight});
        sites_of_.emplace_back(a, b);
    }

    // One turn up on the uphill side; and n also changes by s where the step crosses angle 0 around the centre.
    void WindingGraph::add_steps(const std::vector<int> &orientation)
    {
        for (const SheetPieces::Step &s : pieces_.steps())
        {
            const int oa = orientation[pieces_.piece_of(s.a)], ob = orientation[pieces_.piece_of(s.b)];
            const int up = oa * s.side_a; // +1: b is uphill of a
            if (up == 0 || ob * s.side_b != -up)
                continue; // the two orientations disagree on the slope between these sheets
            const double dphi = phase_[s.b] - phase_[s.a];
            const double across_zero = wrap_angle(TWO_PI * dphi) / TWO_PI - dphi;
            observe(s.a, s.b, up + handedness_ * across_zero, s.weight);
        }
    }

    void WindingGraph::add_links(const SheetSkeleton &skeleton, const std::vector<Peak> &peaks, int centre)
    {
        const auto &nodes = skeleton.nodes();
        for (const SheetPieces::Link &l : pieces_.links())
        {
            double dn = 0;
            if (!l.junction)
                dn = continuous_shift_[l.b] - continuous_shift_[l.a];
            else if (centre >= 0)
            {
                // across the junction or the break, straight from one end to the other
                const double cx = peaks[centre].x, cy = peaks[centre].y;
                const auto &ea = nodes[l.a], &eb = nodes[l.b];
                const double across = wrap_angle(std::atan2(eb.y - cy, eb.x - cx) - std::atan2(ea.y - cy, ea.x - cx));
                dn = handedness_ * (across / TWO_PI - (phase_[l.b] - phase_[l.a]));
            }
            observe(l.a, l.b, dn, l.weight);
        }
    }

    void WindingGraph::add_anchors(const std::vector<Ray> &rays, int centre)
    {
        // without a centre the levels are only relative: one site of the outer face is level 0 (the outer face is not
        // one level all around: past the end of the last turn, it is the turn before), on the largest part of the
        // sheets tied together, so that it is the one labelled
        if (centre < 0)
        {
            const int site = best_outermost();
            if (site >= 0)
                observe(site, -1, 0.0, params_.weight_outermost);
            return;
        }
        // the first sheet around the core is turn 1
        int first = 0;
        for (const Ray &ray : rays)
        {
            if (ray.peak == centre && !ray.crossings.empty())
                observe(pieces_.site_of(first), -1, 1.0, params_.weight_core);
            first += static_cast<int>(ray.crossings.size());
        }
    }

    void WindingGraph::add_spots(const SheetSkeleton &skeleton, const SpotHeights &spots)
    {
        gauge_ = pieces_.pieces();
        const auto &nodes = skeleton.nodes();
        const int every = std::max(params_.spot_every_px, 1);
        for (int s = 0; s < pieces_.sites(); ++s)
        {
            const int node = pieces_.node_of(s);
            if (node >= 0 && nodes[node].depth % every != 0)
                continue;
            const std::optional<int> winding = spots.find(pieces_.position(s).x, pieces_.position(s).y);
            if (!winding)
                continue;
            observations_.push_back({gauge_, pieces_.piece_of(s), *winding - shift_[s], params_.weight_spot});
            sites_of_.emplace_back(s, -1);
        }
    }
} // namespace pw
