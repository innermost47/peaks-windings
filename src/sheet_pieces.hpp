// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "ray_crossings.hpp"
#include "sheet_directions.hpp"
#include "sheet_skeleton.hpp"
#include "voting_solver.hpp"

#include <vector>

namespace pw
{
    // The sheets of one slice as the contour lines of a map, cut into rigid pieces, and what they tell about each
    // other, independently of any centre.
    //
    // The sites are the nodes of the skeleton, then the crossings of the rays that lie on no branch. A piece is a
    // stretch of one branch (at most piece_px long), or such a crossing by itself. Every site has a normal to its sheet,
    // in a sense kept all along its piece (the left of the branch, as numbered). The relations:
    //   step      a probe from a site along its normal, one way or the other, reaching the next sheet: neighbour sheets,
    //             one turn apart; each lies on one side of the other (side_a: the side of a's normal b lies on, side_b
    //             likewise)
    //   along     two pieces in a row on one branch: the same sheet
    //   junction  two branches going on straight through a junction, or across a break: the same sheet
    //   outermost a probe leaving the scroll without meeting a sheet: that side of the site is the outer face
    //
    // The orientation of a piece (+1 / -1) says on which side of it the turns go up (towards the outside of the scroll):
    // +1 on the side of its normal. Read as a contour map, it is the uphill side. It is the same along a sheet, and
    // between two neighbour sheets the slope is one: if b is uphill of a, a is downhill of b.
    class SheetPieces
    {
    public:
        // Weights of the observations, as votes. A probe counts fully when it meets the next sheet straight on (|cos| of
        // the incidence) and no further than usual (the weight falls off beyond probe_gap_tolerance times the median
        // distance probes travel).
        struct Params
        {
            int piece_px = 256;     // longest rigid piece of a branch (px of skeleton)
            int probe_every_px = 8; // px of skeleton between two probes along a branch
            double probe_max_px = 60;
            double probe_gap_tolerance = 2;
            double weight_step = 1;
            double weight_along = 3;
            double weight_junction = 2;
            double junction_max_turn_deg = 35; // two branches go on through a junction when they bend less than this
            double weight_bridge = 2;          // a sheet going on across a break (see SheetSkeleton::bridges)
            double weight_outermost = 1;
        };

        struct Step
        {
            int a, b;           // sites
            int side_a, side_b; // +-1
            double weight;
        };

        // Two sites on one sheet, on different pieces. For a junction or a break, a and b are the nodes where the two
        // branches reach it (`junction` set); `flip` relates the senses of their normals (+1: the same, -1: opposite).
        struct Link
        {
            int a, b;
            bool junction = false;
            int flip = 1;
            double weight;
        };

        struct Outermost
        {
            int site;
            int side; // the side of its normal the scroll ends on
        };

        // `node_of` gives, for every crossing (rays in order, crossings in order along each ray), the skeleton node it
        // is attached to (-1: none). `inside` is the interior of the scroll at full resolution (0 / 1).
        SheetPieces(const std::vector<Ray> &rays, const SheetSkeleton &skeleton, const std::vector<int> &node_of,
                    const SheetDirections &directions, const cv::Mat &inside, const Params &params);

        int sites() const { return static_cast<int>(position_.size()); }
        int pieces() const { return pieces_; }
        int piece_of(int site) const { return piece_of_[site]; }
        // The skeleton node of a site, -1 for a crossing on no branch.
        int node_of(int site) const { return site < nodes_ ? site : -1; }
        const cv::Point2d &position(int site) const { return position_[site]; }
        // The site of a crossing (flat index, rays in order).
        int site_of(int crossing) const { return site_of_crossing_[crossing]; }

        const std::vector<Step> &steps() const { return steps_; }
        const std::vector<Link> &links() const { return links_; }
        const std::vector<Outermost> &outermost() const { return outermost_; }

        // The observations on the orientations of the pieces (anchored on the outer face of the scroll).
        std::vector<Observation> orientation_observations() const;

    private:
        void make_sites(const std::vector<Ray> &rays, const SheetSkeleton &skeleton, const std::vector<int> &node_of,
                        const SheetDirections &directions);
        void cut_pieces(const SheetSkeleton &skeleton);
        void probe(const SheetSkeleton &skeleton, const cv::Mat &inside);
        void link_branches(const SheetSkeleton &skeleton);
        void link_junctions(const SheetSkeleton &skeleton);
        void link_ends(const SheetSkeleton &skeleton, const SheetSkeleton::End &ei, const SheetSkeleton::End &ej,
                       double weight);

        Params params_;
        int nodes_ = 0; // sites [0, nodes_) are the skeleton nodes
        int pieces_ = 0;
        std::vector<cv::Point2d> position_, normal_; // per site; normal (0, 0) where the sense is unknown
        std::vector<int> piece_of_;
        std::vector<int> site_of_crossing_;
        std::vector<Step> steps_;
        std::vector<Link> links_;
        std::vector<Outermost> outermost_;
    };
} // namespace pw
