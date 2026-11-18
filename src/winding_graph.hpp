// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "peaks.hpp"
#include "ray_crossings.hpp"
#include "sheet_pieces.hpp"
#include "sheet_skeleton.hpp"
#include "spot_heights.hpp"
#include "voting_solver.hpp"

#include <utility>
#include <vector>

namespace pw
{
    // The windings of the sheets of one slice, read as the heights of a contour map.
    //
    // The winding n of a site (the number of its turn) is the same all along its sheet and goes up by one from a sheet
    // to the next one on its uphill side (the orientation of the pieces, solved beforehand). That holds everywhere but
    // around the heart of the scroll: there the sheet is a spiral, and following it once around gains one turn. With a
    // core as `centre`, n therefore also changes by s (the direction the scroll is rolled in, +-1) where a sheet crosses
    // the half-line at angle 0 from the core:
    //   n = U_piece + shift,  shift = s * (Theta / 2 pi - phi)
    // Theta being the angle swept around the core along the skeleton and phi the angle of the site (in turns, 0 .. 1).
    // Without a centre (no core in the slice, or a core that is a dome cut across, not a spiral), shift = 0.
    //
    // Observations between pieces: steps (one turn up on the uphill side; dropped where the two orientations
    // disagree), along and junction links (the same sheet), and with a centre the first sheet met by every ray from it
    // (turn 1). Without a centre, one site of the outer face is level 0.
    //
    // Along z, the spot heights of the previous slice: a site lying on one of them is on the same turn. Their own
    // numbering may differ from this slice's by a constant (another centre, none), so they vote on the label of every
    // piece through one more unknown, the gauge: label[piece] = label[gauge] + (spot winding - shift). They thus tie
    // the pieces to one another as they were tied in the previous slice, without imposing its zero.
    class WindingGraph
    {
    public:
        struct Params
        {
            double weight_core = 1;      // per ray from the core: its first crossing is on turn 1
            double weight_outermost = 1; // without a centre: one site of the outer face is level 0
            double max_mismatch = 0.25;  // turns: an observation further than this from a whole number is dropped
            double weight_spot = 1.0;    // per site reading a spot height of the previous slice
            int spot_every_px = 4;       // along the branches, one site in this many reads the spot heights
        };

        // `orientation` is the solved orientation of every piece (+-1).
        WindingGraph(const SheetPieces &pieces, const std::vector<int> &orientation, const SheetSkeleton &skeleton,
                     const std::vector<Ray> &rays, const std::vector<Peak> &peaks, int centre, int handedness,
                     const SpotHeights *spots, const Params &params);

        const std::vector<Observation> &observations() const { return observations_; }
        // the pieces, then the gauge of the spot heights when there are any
        int unknowns() const { return pieces_.pieces() + (gauge_ >= 0 ? 1 : 0); }

        // The winding of a site from the solved labels.
        int winding(int site, const std::vector<int> &labels) const
        {
            return labels[pieces_.piece_of(site)] + shift_[site];
        }
        // The same before rounding: what the angle swept gives exactly.
        double continuous_winding(int site, const std::vector<int> &labels) const
        {
            return labels[pieces_.piece_of(site)] + continuous_shift_[site];
        }
        // The two sites an observation was read from (the second is -1 for an anchor or a spot height).
        const std::pair<int, int> &sites_of(int observation) const { return sites_of_[observation]; }

    private:
        void measure_angles(const SheetSkeleton &skeleton, const std::vector<Peak> &peaks, int centre);
        void add_steps(const std::vector<int> &orientation);
        void add_links(const SheetSkeleton &skeleton, const std::vector<Peak> &peaks, int centre);
        void add_anchors(const std::vector<Ray> &rays, int centre);
        void add_spots(const SheetSkeleton &skeleton, const SpotHeights &spots);
        // the site of the outer face on the most observed group of pieces tied together, -1 for none
        int best_outermost() const;
        // n_b - n_a = dn (b < 0: n_a = dn), as an observation on the labels of their pieces
        void observe(int a, int b, double dn, double weight);

        const SheetPieces &pieces_;
        Params params_;
        int handedness_;
        int gauge_ = -1; // the unknown of the spot heights' numbering, -1 without them
        std::vector<double> phase_;            // per site, around the centre (turns, 0 .. 1); 0 without one
        std::vector<double> continuous_shift_; // per site
        std::vector<int> shift_;               // per site
        std::vector<Observation> observations_;
        std::vector<std::pair<int, int>> sites_of_;
    };
} // namespace pw
