// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "windings.hpp"

#include "numeric.hpp"
#include "parallel.hpp"
#include "ray_crossings.hpp"
#include "scroll_mask.hpp"
#include "sheet_directions.hpp"
#include "sheet_pieces.hpp"
#include "sheet_skeleton.hpp"
#include "summits.hpp"
#include "voting_solver.hpp"
#include "winding_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

namespace pw
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        double seconds_since(Clock::time_point t0)
        {
            return std::chrono::duration<double>(Clock::now() - t0).count();
        }

        // A whole turn off, in the "sigmas" of the pull written for every crossing: beyond the 3 sigmas that
        // unroll_layer leaves out by default.
        constexpr double TURN_IN_SIGMAS = 4.0;

        // One way of counting the turns: around a core (its index among the summits, with the direction the scroll
        // is rolled in), or around none (-1): the observations it gives, their labelling and its tension.
        struct Trial
        {
            int centre = -1;
            int handedness = 1;
            std::unique_ptr<WindingGraph> graph;
            VotingSolver<IntegerShift>::Result solution;
            double tension = std::numeric_limits<double>::infinity();
        };

        // The spiral may turn either way around every core; and a core may be no spiral at all (a dome of a fold cut
        // across), hence also the trial without a centre.
        std::vector<Trial> trials_for(const std::vector<Peak> &peaks)
        {
            std::vector<Trial> trials(1);
            for (int p = 0; p < static_cast<int>(peaks.size()); ++p)
                if (kind_of(peaks[p]) == SummitKind::core)
                    for (int handedness : {1, -1})
                    {
                        trials.emplace_back();
                        trials.back().centre = p;
                        trials.back().handedness = handedness;
                    }
            return trials;
        }

        void run(Trial &trial, const SheetPieces &pieces, const std::vector<int> &orientation,
                 const SheetSkeleton &skeleton, const std::vector<Ray> &rays, const std::vector<Peak> &peaks,
                 const SpotHeights *spots)
        {
            trial.graph = std::make_unique<WindingGraph>(pieces, orientation, skeleton, rays, peaks, trial.centre,
                                                         trial.handedness, spots, WindingGraph::Params{});
            trial.solution = VotingSolver<IntegerShift>().solve(trial.graph->unknowns(), trial.graph->observations());
            trial.tension = trial.solution.tension;
        }

        // The trial leaving the fewest observations unmet. Along z, a trial around the previous centre is kept while
        // its score (1 / (1 + tension)) stays within `keep` of the best.
        int choose(const std::vector<Trial> &trials, const std::vector<Peak> &peaks, const cv::Point2d *previous,
                   double keep = 0.9, double radius = 60.0)
        {
            auto score = [&](int t)
            { return 1.0 / (1.0 + trials[t].tension); };
            int best = 0;
            for (int t = 1; t < static_cast<int>(trials.size()); ++t)
                if (score(t) > score(best))
                    best = t;
            if (!previous)
                return best;
            int near = -1;
            for (int t = 0; t < static_cast<int>(trials.size()); ++t)
            {
                const int c = trials[t].centre;
                if (c >= 0 && std::hypot(peaks[c].x - previous->x, peaks[c].y - previous->y) < radius &&
                    (near < 0 || score(t) > score(near)))
                    near = t;
            }
            return near >= 0 && score(near) >= keep * score(best) ? near : best;
        }

        // The skeleton node every crossing is attached to (rays in order, crossings along each ray), -1 for none.
        std::vector<int> attach_crossings(const std::vector<Ray> &rays, const SheetSkeleton &skeleton, int threads)
        {
            std::vector<int> first(rays.size() + 1, 0);
            for (size_t i = 0; i < rays.size(); ++i)
                first[i + 1] = first[i] + static_cast<int>(rays[i].crossings.size());
            std::vector<int> node_of(first.back(), -1);
            parallel_for(static_cast<int>(rays.size()), threads, [&](int i, int)
                         {
                             for (size_t k = 0; k < rays[i].crossings.size(); ++k)
                                 node_of[first[i] + k] = skeleton.attach(rays[i].crossings[k].x, rays[i].crossings[k].y); });
            return node_of;
        }

        // Every solved site, for the next slice to read.
        SpotHeights spot_heights(const SheetPieces &pieces, const Trial &trial)
        {
            std::vector<SpotHeight> spots;
            spots.reserve(pieces.sites());
            for (int s = 0; s < pieces.sites(); ++s)
                if (trial.solution.reached[pieces.piece_of(s)])
                    spots.push_back({static_cast<float>(pieces.position(s).x), static_cast<float>(pieces.position(s).y),
                                     trial.graph->winding(s, trial.solution.label)});
            return SpotHeights(std::move(spots));
        }

        // The solved crossings, ray by ray.
        void write_lines(const std::vector<Ray> &rays, const std::vector<Peak> &peaks, const SheetSkeleton &skeleton,
                         const SheetPieces &pieces, const Trial &trial, SliceResult &result)
        {
            const WindingGraph &graph = *trial.graph;
            const VotingSolver<IntegerShift>::Result &solution = trial.solution;
            // pull of a piece: the share of the weight of its observations left unmet, as sigmas (a piece most of whose
            // evidence disagrees with its label is beyond what unroll_layer keeps by default)
            std::vector<double> total(pieces.pieces(), 0.0), unmet(pieces.pieces(), 0.0);
            for (size_t i = 0; i < graph.observations().size(); ++i)
            {
                const auto [a, b] = graph.sites_of(static_cast<int>(i));
                for (int s : {a, b})
                    if (s >= 0)
                    {
                        total[pieces.piece_of(s)] += graph.observations()[i].weight;
                        if (solution.residual[i] != 0)
                            unmet[pieces.piece_of(s)] += graph.observations()[i].weight;
                    }
            }
            auto pull = [&](int piece)
            { return total[piece] > 0 ? 2 * TURN_IN_SIGMAS * unmet[piece] / total[piece] : 0.0; };
            int crossing = 0;
            for (const Ray &ray : rays)
            {
                LineOut line;
                line.ray_index = ray.index;
                line.peak_index = ray.peak;
                line.track = peaks[ray.peak].track;
                line.angle = ray.angle;
                for (const Crossing &c : ray.crossings)
                {
                    const int site = pieces.site_of(crossing++);
                    const int piece = pieces.piece_of(site);
                    if (!solution.reached[piece])
                        continue;
                    const int node = pieces.node_of(site);
                    const int fragment = node >= 0 ? skeleton.nodes()[node].branch + 1 : 0;
                    line.crossings.push_back({c.x, c.y, graph.winding(site, solution.label),
                                              graph.continuous_winding(site, solution.label), fragment, pull(piece)});
                }
                if (!line.crossings.empty())
                    result.lines.push_back(std::move(line));
            }
        }
    } // namespace

    SliceResult process_slice(const cv::Mat &img_u8, const Params &params, const cv::Point2d *previous_centre,
                              TrackState *tracks, const SpotHeights *previous_spots)
    {
        CV_Assert(img_u8.type() == CV_8U);
        SliceResult result;
        const int threads = params.threads > 0 ? params.threads : hardware_threads();
        result.n_rays = std::max(params.rays, 1);

        auto t0 = Clock::now();
        const ScrollMasks masks = detect_scroll_masks(img_u8, params);
        result.timings.mask = seconds_since(t0);

        t0 = Clock::now();
        // every summit read is a candidate, so that a tracked summit that lost a few ranks is still found
        const SheetDirections directions(masks.spires_small, masks.mask_small, params.scale, params.direction_sigma_px);
        const std::vector<Peak> candidates = find_summits(directions);
        const std::vector<Peak> peaks =
            select_peaks(candidates, params.peaks, tracks, params.track_radius, params.track_memory);
        result.timings.peaks = seconds_since(t0);
        if (peaks.empty())
            return result;

        t0 = Clock::now();
        const RayParams ray_params{result.n_rays, params.max_radius, params.min_spire, params.min_gap};
        const std::vector<Ray> rays = shoot_rays(masks.spires, masks.inside, peaks, ray_params, threads);
        result.timings.rays = seconds_since(t0);

        t0 = Clock::now();
        const SheetSkeleton skeleton(masks.spires, SheetSkeleton::Params{}, threads);
        const std::vector<int> node_of = attach_crossings(rays, skeleton, threads);
        result.timings.skeleton = seconds_since(t0);

        // which side of every piece of sheet is uphill: the same for every way of counting the turns
        t0 = Clock::now();
        const SheetPieces pieces(rays, skeleton, node_of, directions, masks.inside, SheetPieces::Params{});
        const std::vector<int> orientation =
            VotingSolver<SignFlip>().solve(pieces.pieces(), pieces.orientation_observations()).label;

        // every way of counting the turns is tried, the trials running side by side
        std::vector<Trial> trials = trials_for(peaks);
        const int jobs = std::max(1, std::min({params.centre_jobs, threads, static_cast<int>(trials.size())}));
        parallel_for(static_cast<int>(trials.size()), jobs, [&](int t, int)
                     { run(trials[t], pieces, orientation, skeleton, rays, peaks, previous_spots); });
        const Trial &trial = trials[choose(trials, peaks, previous_centre)];
        result.timings.centre = seconds_since(t0);

        t0 = Clock::now();
        write_lines(rays, peaks, skeleton, pieces, trial, result);
        result.spots = spot_heights(pieces, trial);
        result.has_centre = trial.centre >= 0;
        if (result.has_centre)
        {
            result.cx = peaks[trial.centre].x;
            result.cy = peaks[trial.centre].y;
            result.centre_peak = trial.centre;
            result.centre_track = peaks[trial.centre].track;
            result.handedness = trial.handedness;
        }
        result.peaks = peaks;
        // the score of a summit: that of its best trial (0 for the tip of a fold, never tried)
        result.scores.assign(peaks.size(), 0.0);
        for (const Trial &t : trials)
            if (t.centre >= 0)
                result.scores[t.centre] = std::max(result.scores[t.centre], 1.0 / (1.0 + t.tension));
        result.timings.output = seconds_since(t0);
        return result;
    }
} // namespace pw
