// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "sheet_skeleton.hpp"

#include "numeric.hpp"
#include "thinning.hpp"

#include <opencv2/imgproc.hpp>

#include <bitset>
#include <climits>
#include <cmath>
#include <deque>
#include <map>

namespace pw
{
    namespace
    {
        constexpr int DX[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        constexpr int DY[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
        constexpr int VISITED = INT_MIN; // a pixel of the branch being ordered, reached once already

        // Pixels where three or more lines meet. On a one-pixel-wide skeleton a line has at most 3 set neighbours
        // (a staircase step); more is a junction, as are 3 separate runs of neighbours.
        cv::Mat junction_pixels(const cv::Mat &skeleton)
        {
            cv::Mat junction = cv::Mat::zeros(skeleton.size(), CV_8U);
            const size_t step = skeleton.step;
            for (int y = 1; y + 1 < skeleton.rows; ++y)
            {
                const unsigned char *row = skeleton.ptr<unsigned char>(y);
                unsigned char *out = junction.ptr<unsigned char>(y);
                for (int x = 1; x + 1 < skeleton.cols; ++x)
                {
                    if (!row[x])
                        continue;
                    const int code = neighbour_code(row + x, step);
                    out[x] = crossing_number(code) >= 3 || std::bitset<8>(code).count() >= 4;
                }
            }
            return junction;
        }
    } // namespace

    SheetSkeleton::SheetSkeleton(const cv::Mat &spires, const Params &params, int threads) : params_(params)
    {
        CV_Assert(spires.type() == CV_8U);
        const cv::Rect bounds = cv::boundingRect(spires);
        if (bounds.area() == 0)
            return;
        // one free pixel around the spires: thinning keeps the outermost pixels of its input empty
        roi_ = cv::Rect(bounds.x - 2, bounds.y - 2, bounds.width + 4, bounds.height + 4) &
               cv::Rect(0, 0, spires.cols, spires.rows);
        cv::Mat skeleton = thin_lines(spires(roi_), threads);
        cv::Mat junction_zone;
        cv::dilate(junction_pixels(skeleton), junction_zone, cv::Mat::ones(3, 3, CV_8U));
        skeleton.setTo(0, junction_zone);
        label_branches(skeleton);
        find_junctions(junction_zone);
        find_bridges(junction_zone);
    }

    void SheetSkeleton::label_branches(const cv::Mat &skeleton)
    {
        cv::Mat stats, centroids;
        cv::connectedComponentsWithStats(skeleton, node_map_, stats, centroids, 8, CV_32S);
        // components are marked by their negated label until they are ordered (node index + 1, positive)
        node_map_ = -node_map_;
        for (int y = 0; y < node_map_.rows; ++y)
        {
            int *row = node_map_.ptr<int>(y);
            for (int x = 0; x < node_map_.cols; ++x)
            {
                if (row[x] >= 0)
                    continue;
                const int area = stats.at<int>(-row[x], cv::CC_STAT_AREA);
                if (area < params_.min_branch_px)
                    erase_component(x, y);
                else
                    order_branch(x, y);
            }
        }
    }

    void SheetSkeleton::erase_component(int seed_x, int seed_y)
    {
        const int label = node_map_.at<int>(seed_y, seed_x);
        std::vector<cv::Point> stack{{seed_x, seed_y}};
        node_map_.at<int>(seed_y, seed_x) = 0;
        while (!stack.empty())
        {
            const cv::Point p = stack.back();
            stack.pop_back();
            for (int k = 0; k < 8; ++k)
            {
                const int x = p.x + DX[k], y = p.y + DY[k];
                if (x < 0 || y < 0 || x >= node_map_.cols || y >= node_map_.rows || node_map_.at<int>(y, x) != label)
                    continue;
                node_map_.at<int>(y, x) = 0;
                stack.push_back({x, y});
            }
        }
    }

    // Two breadth-first walks: the first from the seed finds the far end of the branch, the second numbers the nodes
    // from that end (so that depth runs along the sheet).
    void SheetSkeleton::order_branch(int seed_x, int seed_y)
    {
        const int w = node_map_.cols, h = node_map_.rows;
        auto walk = [&](int sx, int sy, auto member, auto visit)
        {
            std::deque<cv::Point> queue{{sx, sy}};
            visit(sx, sy, -1, -1);
            cv::Point last(sx, sy);
            while (!queue.empty())
            {
                const cv::Point p = queue.front();
                queue.pop_front();
                last = p;
                for (int k = 0; k < 8; ++k)
                {
                    const int x = p.x + DX[k], y = p.y + DY[k];
                    if (x < 0 || y < 0 || x >= w || y >= h || !member(node_map_.at<int>(y, x)))
                        continue;
                    visit(x, y, p.x, p.y);
                    queue.push_back({x, y});
                }
            }
            return last;
        };
        const cv::Point far_end = walk(
            seed_x, seed_y, [](int v)
            { return v < 0 && v != VISITED; },
            [&](int x, int y, int, int)
            { node_map_.at<int>(y, x) = VISITED; });

        Branch branch;
        branch.begin = static_cast<int>(nodes_.size());
        const int id = static_cast<int>(branches_.size());
        walk(
            far_end.x, far_end.y, [](int v)
            { return v == VISITED; },
            [&](int x, int y, int px, int py)
            {
                Node node;
                node.x = x + roi_.x;
                node.y = y + roi_.y;
                node.branch = id;
                if (px >= 0)
                {
                    node.parent = node_map_.at<int>(py, px) - 1;
                    node.depth = nodes_[node.parent].depth + 1;
                }
                node_map_.at<int>(y, x) = static_cast<int>(nodes_.size()) + 1;
                nodes_.push_back(node);
            });
        branch.end = static_cast<int>(nodes_.size());
        branches_.push_back(branch);
    }

    void SheetSkeleton::find_junctions(const cv::Mat &junction_zone)
    {
        cv::Mat clusters;
        const int n_clusters = cv::connectedComponents(junction_zone, clusters, 8, CV_32S);
        // for every cluster, the branches touching it, each by its node nearest to the cluster
        std::vector<std::map<int, int>> touching(n_clusters); // cluster -> {branch: node}
        std::vector<cv::Point2d> sum(n_clusters);
        std::vector<int> count(n_clusters, 0);
        std::vector<cv::Point> zone;
        cv::findNonZero(junction_zone, zone);
        for (const cv::Point &p : zone)
        {
            const int c = clusters.at<int>(p);
            sum[c] += cv::Point2d(p.x + roi_.x, p.y + roi_.y);
            ++count[c];
            for (int k = 0; k < 8; ++k)
            {
                const int x = p.x + DX[k], y = p.y + DY[k];
                if (x < 0 || y < 0 || x >= node_map_.cols || y >= node_map_.rows)
                    continue;
                const int node = node_map_.at<int>(y, x) - 1;
                if (node >= 0)
                    touching[c].emplace(nodes_[node].branch, node);
            }
        }
        for (int c = 1; c < n_clusters; ++c)
        {
            if (touching[c].size() < 2)
                continue;
            const cv::Point2d centre = sum[c] / count[c];
            Junction junction;
            for (const auto &[branch, node] : touching[c])
                junction.ends.push_back(end_towards(node, centre.x, centre.y));
            junctions_.push_back(std::move(junction));
        }
    }

    // The direction a branch arrives in at `node`: from the mean of its nodes within end_tangent_px to the junction.
    SheetSkeleton::End SheetSkeleton::end_towards(int node, double jx, double jy) const
    {
        const Node &end = nodes_[node];
        const int r = params_.end_tangent_px;
        double mx = 0, my = 0;
        int n = 0;
        for (int y = end.y - r; y <= end.y + r; ++y)
            for (int x = end.x - r; x <= end.x + r; ++x)
            {
                const int lx = x - roi_.x, ly = y - roi_.y;
                if (lx < 0 || ly < 0 || lx >= node_map_.cols || ly >= node_map_.rows)
                    continue;
                const int other = node_map_.at<int>(ly, lx) - 1;
                if (other >= 0 && nodes_[other].branch == end.branch)
                    mx += x, my += y, ++n;
            }
        End out;
        out.node = node;
        double dx = jx - mx / n, dy = jy - my / n;
        const double norm = std::hypot(dx, dy);
        if (norm > 0)
            out.dx = dx / norm, out.dy = dy / norm;
        return out;
    }

    // A sheet broken across a gap leaves two free ends (ends of branches touching no junction) that face each other
    // on one line. Each free end is bridged to the one that suits it best, when the choice is mutual.
    void SheetSkeleton::find_bridges(const cv::Mat &junction_zone)
    {
        std::vector<End> ends;
        for (int i = 0; i < static_cast<int>(nodes_.size()); ++i)
        {
            const Node &n = nodes_[i];
            const int lx = n.x - roi_.x, ly = n.y - roi_.y;
            int neighbours = 0;
            bool at_junction = false;
            for (int k = 0; k < 8; ++k)
            {
                const int x = lx + DX[k], y = ly + DY[k];
                if (x < 0 || y < 0 || x >= node_map_.cols || y >= node_map_.rows)
                    continue;
                at_junction = at_junction || junction_zone.at<unsigned char>(y, x) != 0;
                const int other = node_map_.at<int>(y, x) - 1;
                neighbours += other >= 0 && nodes_[other].branch == n.branch;
            }
            if (neighbours == 1 && !at_junction)
                ends.push_back(end_towards(i, n.x, n.y)); // pointing out of the branch, into the gap
        }

        const double reach = params_.bridge_max_px;
        const double facing = std::cos(params_.bridge_max_turn_deg * PI / 180.0);
        std::map<std::pair<int, int>, std::vector<int>> cells;
        auto cell_of = [&](const Node &n)
        { return std::make_pair(static_cast<int>(std::floor(n.x / reach)), static_cast<int>(std::floor(n.y / reach))); };
        for (int i = 0; i < static_cast<int>(ends.size()); ++i)
            cells[cell_of(nodes_[ends[i].node])].push_back(i);

        std::vector<int> partner(ends.size(), -1);
        for (int i = 0; i < static_cast<int>(ends.size()); ++i)
        {
            const End &ei = ends[i];
            const Node &a = nodes_[ei.node];
            const auto [cx, cy] = cell_of(a);
            double best = reach;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const auto it = cells.find({cx + dx, cy + dy});
                    if (it == cells.end())
                        continue;
                    for (int j : it->second)
                    {
                        const End &ej = ends[j];
                        const Node &b = nodes_[ej.node];
                        if (b.branch == a.branch)
                            continue;
                        const double vx = b.x - a.x, vy = b.y - a.y, d = std::hypot(vx, vy);
                        if (d == 0 || d >= best)
                            continue;
                        const bool face = (ei.dx * vx + ei.dy * vy) >= facing * d && -(ej.dx * vx + ej.dy * vy) >= facing * d;
                        const bool in_line = std::abs(ei.dx * vy - ei.dy * vx) <= params_.bridge_max_lateral_px &&
                                             std::abs(ej.dx * vy - ej.dy * vx) <= params_.bridge_max_lateral_px;
                        if (face && in_line && clear_between(a, b))
                            best = d, partner[i] = j;
                    }
                }
        }
        for (int i = 0; i < static_cast<int>(ends.size()); ++i)
            if (partner[i] > i && partner[partner[i]] == i)
                bridges_.emplace_back(ends[i], ends[partner[i]]);
    }

    // Whether the segment between two nodes crosses no branch but theirs.
    bool SheetSkeleton::clear_between(const Node &a, const Node &b) const
    {
        const double length = std::hypot(b.x - a.x, b.y - a.y);
        const int steps = static_cast<int>(std::ceil(length));
        for (int s = 1; s < steps; ++s)
        {
            const double t = static_cast<double>(s) / steps;
            const int x = round_half_even(a.x + t * (b.x - a.x)) - roi_.x, y = round_half_even(a.y + t * (b.y - a.y)) - roi_.y;
            for (int k = -1; k < 4; ++k) // the pixel and its 4 neighbours: a diagonal line slips between pixels
            {
                const int px = x + (k < 0 ? 0 : DX[2 * k]), py = y + (k < 0 ? 0 : DY[2 * k]);
                if (px < 0 || py < 0 || px >= node_map_.cols || py >= node_map_.rows)
                    continue;
                const int other = node_map_.at<int>(py, px) - 1;
                if (other >= 0 && nodes_[other].branch != a.branch && nodes_[other].branch != b.branch)
                    return false;
            }
        }
        return true;
    }

    int SheetSkeleton::node_at(int x, int y) const
    {
        const int lx = x - roi_.x, ly = y - roi_.y;
        for (int k = -1; k < 4; ++k)
        {
            const int px = lx + (k < 0 ? 0 : DX[2 * k]), py = ly + (k < 0 ? 0 : DY[2 * k]);
            if (px < 0 || py < 0 || px >= node_map_.cols || py >= node_map_.rows)
                continue;
            const int node = node_map_.at<int>(py, px) - 1;
            if (node >= 0)
                return node;
        }
        return -1;
    }

    int SheetSkeleton::attach(double x, double y) const
    {
        if (node_map_.empty())
            return -1;
        const int r = static_cast<int>(std::ceil(params_.snap_radius));
        const int cx = round_half_even(x) - roi_.x, cy = round_half_even(y) - roi_.y;
        int best = -1;
        double best_d2 = params_.snap_radius * params_.snap_radius;
        for (int ly = std::max(cy - r, 0); ly <= std::min(cy + r, node_map_.rows - 1); ++ly)
        {
            const int *row = node_map_.ptr<int>(ly);
            for (int lx = std::max(cx - r, 0); lx <= std::min(cx + r, node_map_.cols - 1); ++lx)
            {
                if (row[lx] <= 0)
                    continue;
                const double dx = lx + roi_.x - x, dy = ly + roi_.y - y, d2 = dx * dx + dy * dy;
                if (d2 <= best_d2)
                {
                    best_d2 = d2;
                    best = row[lx] - 1;
                }
            }
        }
        return best;
    }

    cv::Point2d SheetSkeleton::tangent(int node, int span) const
    {
        int ancestor = node;
        for (int k = 0; k < span && nodes_[ancestor].parent >= 0; ++k)
            ancestor = nodes_[ancestor].parent;
        const cv::Point2d d(nodes_[node].x - nodes_[ancestor].x, nodes_[node].y - nodes_[ancestor].y);
        const double norm = std::hypot(d.x, d.y);
        return norm > 0 ? d / norm : cv::Point2d(0, 0);
    }

    std::vector<double> SheetSkeleton::unwrapped_angles(double cx, double cy) const
    {
        std::vector<double> theta(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            const Node &n = nodes_[i];
            const double a = std::atan2(n.y - cy, n.x - cx);
            theta[i] = n.parent < 0 ? a : theta[n.parent] + wrap_angle(a - std::atan2(nodes_[n.parent].y - cy,
                                                                                        nodes_[n.parent].x - cx));
        }
        return theta;
    }
} // namespace pw
