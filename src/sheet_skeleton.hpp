// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace pw
{
    // The centre lines of the sheets of one slice, cut wherever two sheets touch.
    //
    // The spires are thinned to one-pixel lines; every pixel where three or more lines meet (a junction: sheets
    // touching or a sheet splitting) is removed together with its neighbours. What is left falls apart into branches,
    // each of them the centre line of one single sheet. The pixels of a branch ("nodes") are numbered in breadth-first
    // order from one of its ends, so that every node comes after its parent: anything integrated along a branch (the
    // angle swept around a centre, the arc length) is one pass over its nodes.
    class SheetSkeleton
    {
    public:
        struct Params
        {
            int min_branch_px = 6;   // shorter branches are specks or spurs, and are dropped
            double snap_radius = 6;  // full-res px: how far from a branch a point may lie to be attached to it
            int end_tangent_px = 10; // full-res px: length of branch over which the direction of an end is measured
            // a break in a sheet: two free ends (on no junction) facing each other at most bridge_max_px apart, both
            // pointing at the other within bridge_max_turn_deg, the line of one passing within bridge_max_lateral_px
            // of the other, and no other branch in between
            double bridge_max_px = 30;
            double bridge_max_turn_deg = 30;
            double bridge_max_lateral_px = 3;
        };

        struct Node
        {
            int x = 0, y = 0;
            int branch = 0;
            int parent = -1; // global index of the parent node, -1 for the root of the branch
            int depth = 0;   // arc length from the root, in steps
        };

        struct Branch
        {
            int begin = 0, end = 0; // node range [begin, end)
        };

        // Where a branch runs into a junction or into a break: its last node there and the direction it arrives in
        // (unit vector pointing from the branch into the junction or the gap).
        struct End
        {
            int node = -1;
            double dx = 0, dy = 0;
        };

        // The branches that meet at one junction (a cluster of removed pixels).
        struct Junction
        {
            std::vector<End> ends;
        };

        SheetSkeleton(const cv::Mat &spires, const Params &params, int threads);

        const std::vector<Node> &nodes() const { return nodes_; }
        const std::vector<Branch> &branches() const { return branches_; }
        const std::vector<Junction> &junctions() const { return junctions_; }
        // The breaks of the sheets: pairs of free ends that go on each other across a gap.
        const std::vector<std::pair<End, End>> &bridges() const { return bridges_; }

        // The node nearest to (x, y) within the snap radius, -1 when no branch is that close.
        int attach(double x, double y) const;

        // The node on pixel (x, y) or on one of its 4 neighbours (a line one pixel wide is never stepped over), -1 for
        // none.
        int node_at(int x, int y) const;

        // Direction of the sheet at a node (unit vector), from the node to its ancestor `span` steps back along the
        // branch; (0, 0) at the root of a branch.
        cv::Point2d tangent(int node, int span = 5) const;

        // The angle swept around (cx, cy) along every branch, from its root, for every node (radians, unwrapped: a
        // branch going once around the centre ends 2 pi away from where it started).
        std::vector<double> unwrapped_angles(double cx, double cy) const;

    private:
        void label_branches(const cv::Mat &skeleton);
        void order_branch(int seed_x, int seed_y);
        void erase_component(int seed_x, int seed_y);
        void find_junctions(const cv::Mat &junction_zone);
        void find_bridges(const cv::Mat &junction_zone);
        bool clear_between(const Node &a, const Node &b) const;
        End end_towards(int node, double jx, double jy) const;

        Params params_;
        cv::Rect roi_;           // region of the slice the maps cover
        cv::Mat node_map_;       // CV_32S over roi_: global node index + 1, 0 where there is none
        std::vector<Node> nodes_;
        std::vector<Branch> branches_;
        std::vector<Junction> junctions_;
        std::vector<std::pair<End, End>> bridges_;
    };
} // namespace pw
