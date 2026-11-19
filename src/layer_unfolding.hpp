// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "layers.hpp"
#include "point_grid.hpp"
#include "rail.hpp"

#include <deque>
#include <limits>
#include <map>
#include <vector>

namespace pw
{
    // One layer flattened along its sheet, as an atlas: in every slice, the position u along the sheet (voxels) of every
    // point of its traces, within a region.
    //
    // A layer is a level of the contour map: it may be several separate lines in one slice (one around every summit of
    // that height), each of them a region with its own u. In one slice the line of a region comes as traces (stretches
    // of skeleton between junctions and breaks), each of them rigid along itself: u = offset + direction * arc length.
    // Slice after slice, a trace is
    //   inherited  when it lies on the sheet of the previous slice: region, offset and direction are fitted to what it
    //              finds there;
    //   joined     when it goes on from a free end of a placed trace, across a junction or a break (a short gap, no
    //              sharp turn): it carries on its region and its u across the gap;
    //   seeded     when it is long and none of the above: it starts a new region, at u = 0.
    // Traces pulled too far by the labelling (their turn is in doubt) are left out. Once every slice is given, the regions
    // are laid side by side in the image, the largest first, the scraps left out.
    class LayerUnfolding
    {
    public:
        struct Params
        {
            double max_pull = 3.0;        // sigmas: traces pulled further are left out (0: every trace is kept)
            double min_trace_px = 20;     // shorter traces are left out
            double inherit_radius_px = 4; // a point finds the sheet of the previous slice within this distance
            int fill_z = 0;               // slices: an unknown column borrows the sheet of a slice this near (0: never)
            int memory_slices = 1;       // slices looked back at to find the sheet (1: the previous one only)
            double min_inherited = 0.3;   // share of the points of a trace that must find it to inherit
            double max_misfit_px = 8;     // median distance between the u found and the rigid fit of the trace
            double max_gap_px = 60;       // ends further apart are not joined
            double max_turn_deg = 45;     // nor ends whose directions turn more than this
            double end_tangent_px = 10;   // length of trace over which the direction of an end is measured
            double fill_px = 30;          // holes along the sheet up to this long are bridged straight
            double min_seed_px = 150;     // shorter traces never start a region
            double min_region_share = 0.02; // regions smaller than this share of the largest are left out (area)
            double region_gap_px = 40;    // blank columns between two regions of the image
        };

        struct Stats
        {
            long slices = 0, inherited = 0, joined = 0, seeded = 0, dropped = 0;
            long regions_kept = 0; // after lay_out
            // the traces left out, by reason (they sum to `dropped`), and the ones offered
            long offered = 0, pulled = 0, too_short = 0, unplaced = 0;
            long regions_total = 0; // before lay_out
        };

        // What a region amounts to, for diagnosis.
        struct RegionSummary
        {
            double width = 0; // u extent (voxels)
            double area = 0;
            int z_first = 0, z_last = 0;
            bool kept = false; // after lay_out
        };

        explicit LayerUnfolding(const Params &params) : params_(params) {}

        // The `top` largest regions by area, the largest first.
        std::vector<RegionSummary> largest_regions(size_t top) const;
        // The total area of all the regions, and of the kept ones (after lay_out).
        double total_area() const;
        double kept_area() const;

        // The traces of the layer in the next slice (slices are given in increasing z).
        void add_slice(int z, const std::vector<Trace> &traces);

        // Lays the regions side by side, the largest first, leaving out the scraps; once, after the last slice.
        void lay_out();

        bool empty() const { return slices_.empty(); }
        int z_first() const { return slices_.begin()->first; }
        int z_last() const { return slices_.rbegin()->first; }
        // The largest x and y of the traces placed (voxels).
        double x_max() const { return x_max_; }
        double y_max() const { return y_max_; }
        // After lay_out: the positions of the image run from 0 to width.
        double width() const { return width_; }
        const Stats &stats() const { return stats_; }

        // After lay_out: the sheet in slice z at the positions u of the image (increasing, evenly spaced), NaN where it
        // is unknown.
        // With `fill_z` and params.fill_z > 0, the columns unknown in slice z take the sheet of the nearest slice (at
        // most params.fill_z away) that knows them.
        Rail rail(int z, const std::vector<double> &u, bool fill_z = true) const;

        // After lay_out: the mean share of the columns u that a slice's rail covers (finite), over the slices placed.
        double coverage(const std::vector<double> &u, bool fill_z = true) const;

    private:
        // A trace: its points and the arc length at each of them.
        struct Polyline
        {
            std::vector<float> x, y;
            std::vector<double> s;
            double length() const { return s.back(); }
        };
        // A placed trace: its region, and the u of every point (monotonic along it).
        struct Placed : Polyline
        {
            int region = 0;
            std::vector<float> u;
        };
        // What a point of the previous slice tells: where it is on the sheet.
        struct Spot
        {
            int region = 0;
            float u = 0;
        };
        struct Region
        {
            double u_min = std::numeric_limits<double>::infinity();
            double u_max = -std::numeric_limits<double>::infinity();
            double area = 0;     // the length of its traces summed over the slices
            double offset = -1;  // after lay_out: where its u = u_min lies in the image (-1: left out)
            int z_first = std::numeric_limits<int>::max(); // the slices it is present in
            int z_last = std::numeric_limits<int>::min();
        };
        // Where a trace ends: its last point and the direction it leaves in (unit vector pointing out of the trace).
        struct End
        {
            double x = 0, y = 0, dx = 0, dy = 0;
        };

        // The rail of slice z from its own traces only.
        Rail slice_rail(int z, const std::vector<double> &u) const;
        static Polyline polyline(const Trace &trace);
        static Placed place(const Polyline &line, int region, double offset, int direction);
        End end_of(const Polyline &line, bool last) const;
        void inherit(const std::vector<Polyline> &candidates, std::vector<char> &done, std::vector<Placed> &placed);
        void join(const std::vector<Polyline> &candidates, std::vector<char> &done, std::vector<Placed> &placed);
        void seed(const std::vector<Polyline> &candidates, std::vector<char> &done, std::vector<Placed> &placed);

        Params params_;
        std::map<int, std::vector<Placed>> slices_; // z -> the placed traces, the first ones prevailing where they overlap
        std::vector<Region> regions_;
        // The last slices placed, the newest first (at most params_.memory_slices): a point finds the sheet in the newest
        // one that holds it.
        std::deque<PointGrid<Spot>> memory_;
        const Spot *recall(double x, double y) const;
        double width_ = 0;
        double x_max_ = 0, y_max_ = 0;
        Stats stats_;
    };
} // namespace pw
