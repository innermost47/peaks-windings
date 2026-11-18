// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "peaks.hpp"
#include "spot_heights.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace pw
{
    struct Params
    {
        double scale = 0.3;         // downsampling for the mask and the relief only
        int peaks = 5;              // summits kept
        int rays = 360;             // rays of every summit (fixed angles: ray k keeps its direction from slice to slice)
        double max_radius = 0.0;    // 0: the size of the slice
        int min_spire = 2;          // px of spire for a crossing
        int min_gap = 3;            // px of void before a new crossing
        double frame_px = 30.0;     // full-res px: thicker than this is the frame (case)
        double close_px = 70.0;     // full-res px: gaps closed to find the interior
        double direction_sigma_px = 40.0; // full-res px: the direction of the sheets is smoothed over this
        int threads = 0;            // 0: all hardware threads
        int centre_jobs = 4;        // candidate centres tried at the same time
        double track_radius = 30.0; // full-res px: a summit found this close to a tracked one is the same summit
        int track_memory = 3;       // slices a tracked summit may stay unseen before its track ends
    };

    struct CrossingOut
    {
        double x = 0, y = 0;
        int winding_label = 0;
        double u_continuous = 0; // the solved winding before rounding
        int fragment = 0;        // the skeleton branch (sheet piece) the crossing lies on, + 1; 0 for none
        double pull = 0;         // largest residual (in sigmas) of the observations on this crossing
    };

    struct LineOut
    {
        int ray_index = 0;  // index of the ray around its summit (0 .. n_rays - 1), the same angle in every slice
        int peak_index = 0; // rank of the summit in this slice (0: the highest)
        int track = 0;      // track number of the summit
        double angle = 0;
        std::vector<CrossingOut> crossings;
    };

    struct Timings
    {
        double mask = 0, skeleton = 0, peaks = 0, rays = 0, centre = 0, output = 0;
    };

    struct SliceResult
    {
        std::vector<LineOut> lines;
        bool has_centre = false;
        double cx = 0, cy = 0;
        int centre_peak = -1;
        int centre_track = -1;
        int handedness = 1; // +1: the distance to the centre grows with the angle along the sheets; -1: against it
        int n_rays = 0;
        std::vector<Peak> peaks;
        std::vector<double> scores;
        SpotHeights spots; // every solved sheet point, for the next slice
        Timings timings;
    };

    // Percentile stretch (1 % .. 99 %) of a raw slice to 8 bits, as in the Python version.
    cv::Mat normalise(const cv::Mat &slice);

    // One slice (8-bit, single channel). `previous_centre` may be null. With `tracks`, the summits are matched to
    // those of the previous slices (and the state is updated); without, they are numbered by height. With
    // `previous_spots` (the `spots` of the previous slice), the sheets keep the turns they had there.
    SliceResult process_slice(const cv::Mat &img_u8, const Params &params, const cv::Point2d *previous_centre,
                              TrackState *tracks = nullptr, const SpotHeights *previous_spots = nullptr);

    // Compact JSON, same layout as the Python version. z < 0: "z" fields are null and no top-level "z".
    std::string to_json(const SliceResult &result, long long z);
} // namespace pw
