// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "sheet_pieces.hpp"

#include "numeric.hpp"

#include <algorithm>
#include <cmath>

namespace pw
{
    namespace
    {
        int sign_of(double v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }
        bool is_zero(const cv::Point2d &v) { return v.x == 0 && v.y == 0; }
    } // namespace

    SheetPieces::SheetPieces(const std::vector<Ray> &rays, const SheetSkeleton &skeleton,
                             const std::vector<int> &node_of, const SheetDirections &directions, const cv::Mat &inside,
                             const Params &params)
        : params_(params)
    {
        make_sites(rays, skeleton, node_of, directions);
        cut_pieces(skeleton);
        probe(skeleton, inside);
        link_branches(skeleton);
        link_junctions(skeleton);
    }

    void SheetPieces::make_sites(const std::vector<Ray> &rays, const SheetSkeleton &skeleton,
                                 const std::vector<int> &node_of, const SheetDirections &directions)
    {
        const auto &nodes = skeleton.nodes();
        nodes_ = static_cast<int>(nodes.size());
        position_.reserve(nodes.size());
        normal_.reserve(nodes.size());
        for (int i = 0; i < nodes_; ++i)
        {
            position_.emplace_back(nodes[i].x, nodes[i].y);
            const cv::Point2d t = skeleton.tangent(i); // (0, 0) at the root of a branch: no sense along it
            normal_.emplace_back(-t.y, t.x);
        }
        // a crossing on a branch is that node; any other is a site of its own, its normal from the sheet directions
        int crossing = 0;
        for (const Ray &ray : rays)
            for (const Crossing &c : ray.crossings)
            {
                const int node = node_of[crossing++];
                if (node >= 0 && !is_zero(normal_[node]))
                {
                    site_of_crossing_.push_back(node);
                    continue;
                }
                site_of_crossing_.push_back(sites());
                position_.emplace_back(c.x, c.y);
                const cv::Point2d t = directions.tangent(c.x, c.y);
                normal_.emplace_back(-t.y, t.x);
            }
    }

    void SheetPieces::cut_pieces(const SheetSkeleton &skeleton)
    {
        const auto &nodes = skeleton.nodes();
        const int length = std::max(params_.piece_px, 1);
        piece_of_.assign(sites(), -1);
        for (const SheetSkeleton::Branch &branch : skeleton.branches())
        {
            int deepest = 0;
            for (int i = branch.begin; i < branch.end; ++i)
                deepest = std::max(deepest, nodes[i].depth);
            std::vector<int> piece_of_bin(deepest / length + 1, -1);
            for (int i = branch.begin; i < branch.end; ++i)
            {
                int &piece = piece_of_bin[nodes[i].depth / length];
                if (piece < 0)
                    piece = pieces_++;
                piece_of_[i] = piece;
            }
        }
        for (int s = nodes_; s < sites(); ++s)
            piece_of_[s] = pieces_++;
    }

    // From sites spread along every branch (and from every crossing on no branch), a probe goes along the normal each
    // way: the first sheet it meets is the neighbour on that side; leaving the scroll first means that side is the
    // outer face.
    void SheetPieces::probe(const SheetSkeleton &skeleton, const cv::Mat &inside)
    {
        const auto &nodes = skeleton.nodes();
        struct Hit
        {
            int a, b, side_a;
            double distance, incidence;
        };
        std::vector<Hit> hits;
        auto launch = [&](int a)
        {
            if (is_zero(normal_[a]))
                return;
            const int own_branch = a < nodes_ ? nodes[a].branch : -1;
            for (int side : {1, -1})
            {
                const cv::Point2d direction = normal_[a] * side;
                for (double t = 1.5; t <= params_.probe_max_px; t += 0.5)
                {
                    const int x = round_half_even(position_[a].x + t * direction.x);
                    const int y = round_half_even(position_[a].y + t * direction.y);
                    if (x < 0 || y < 0 || x >= inside.cols || y >= inside.rows || !inside.at<unsigned char>(y, x))
                    {
                        outermost_.push_back({a, side});
                        break;
                    }
                    const int b = skeleton.node_at(x, y);
                    if (b < 0)
                        continue;
                    if (nodes[b].branch == own_branch)
                    {
                        if (t < 3)
                            continue; // still on its own sheet
                        break;        // its own sheet folded back: the branch already ties them
                    }
                    if (!is_zero(normal_[b]))
                        hits.push_back({a, b, side, t, std::abs(normal_[b].dot(direction))});
                    break;
                }
            }
        };
        const int every = std::max(params_.probe_every_px, 1);
        for (int i = 0; i < nodes_; ++i)
            if (nodes[i].depth % every == every / 2)
                launch(i);
        for (int s = nodes_; s < sites(); ++s)
            launch(s);

        // the usual distance between neighbour sheets
        std::vector<double> distances;
        for (const Hit &h : hits)
            distances.push_back(h.distance);
        double usual = 1;
        if (!distances.empty())
        {
            std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
            usual = std::max(distances[distances.size() / 2], 1.0);
        }
        const double tolerance = params_.probe_gap_tolerance * usual;
        for (const Hit &h : hits)
        {
            // a lies on the side of b's normal the probe came from
            const int side_b = sign_of(-normal_[h.b].dot(normal_[h.a] * h.side_a));
            if (side_b == 0)
                continue;
            const double gap = h.distance / tolerance;
            steps_.push_back({h.a, h.b, h.side_a, side_b, params_.weight_step * h.incidence / (1.0 + gap * gap * gap * gap)});
        }
    }

    void SheetPieces::link_branches(const SheetSkeleton &skeleton)
    {
        const auto &nodes = skeleton.nodes();
        for (int i = 0; i < nodes_; ++i)
        {
            const int parent = nodes[i].parent;
            if (parent >= 0 && piece_of_[parent] != piece_of_[i] && !is_zero(normal_[parent]) && !is_zero(normal_[i]))
                links_.push_back({parent, i, false, 1, params_.weight_along});
        }
    }

    void SheetPieces::link_junctions(const SheetSkeleton &skeleton)
    {
        const double straight = -std::cos(params_.junction_max_turn_deg * PI / 180.0);
        for (const SheetSkeleton::Junction &junction : skeleton.junctions())
        {
            struct Pair
            {
                double cosine;
                int i, j;
            };
            std::vector<Pair> pairs;
            const auto &ends = junction.ends;
            for (int i = 0; i < static_cast<int>(ends.size()); ++i)
                for (int j = i + 1; j < static_cast<int>(ends.size()); ++j)
                {
                    const double cosine = ends[i].dx * ends[j].dx + ends[i].dy * ends[j].dy;
                    if (cosine < straight) // they arrive from opposite sides: one sheet going on
                        pairs.push_back({cosine, i, j});
                }
            std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b)
                      { return a.cosine < b.cosine; });
            std::vector<char> used(ends.size(), 0);
            for (const Pair &p : pairs)
            {
                if (used[p.i] || used[p.j])
                    continue;
                used[p.i] = used[p.j] = 1;
                link_ends(skeleton, ends[p.i], ends[p.j], params_.weight_junction);
            }
        }
        for (const auto &[ei, ej] : skeleton.bridges())
            link_ends(skeleton, ei, ej, params_.weight_bridge);
    }

    // One sheet going on from the end of one branch to the end of another (through a junction or across a break).
    void SheetPieces::link_ends(const SheetSkeleton &skeleton, const SheetSkeleton::End &ei,
                                const SheetSkeleton::End &ej, double weight)
    {
        const cv::Point2d ti = skeleton.tangent(ei.node), tj = skeleton.tangent(ej.node);
        if (is_zero(ti) || is_zero(tj))
            return;
        // going from branch i to branch j: out of i at its end, into j at its end
        const int along_i = sign_of(ti.x * ei.dx + ti.y * ei.dy);
        const int along_j = sign_of(-(tj.x * ej.dx + tj.y * ej.dy));
        if (along_i == 0 || along_j == 0)
            return;
        links_.push_back({ei.node, ej.node, true, along_i * along_j, weight});
    }

    std::vector<Observation> SheetPieces::orientation_observations() const
    {
        std::vector<Observation> observations;
        // b uphill of a (o_a side_a = +1) means a downhill of b (o_b side_b = -1): o_b = -o_a side_a side_b
        for (const Step &s : steps_)
            if (piece_of_[s.a] != piece_of_[s.b])
                observations.push_back({piece_of_[s.a], piece_of_[s.b], -s.side_a * s.side_b, s.weight});
        for (const Link &l : links_)
            if (piece_of_[l.a] != piece_of_[l.b])
                observations.push_back({piece_of_[l.a], piece_of_[l.b], l.flip, l.weight});
        // the turns go up towards the outside: on the outer face, uphill is the side the scroll ends on
        for (const Outermost &o : outermost_)
            observations.push_back({piece_of_[o.site], -1, o.side, params_.weight_outermost});
        return observations;
    }
} // namespace pw
