// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "sheet_traces.hpp"

#include <algorithm>

namespace pw
{
    namespace
    {
        // The arms of the breadth-first trees of the branches: from the deepest leaf up to the root first, then from
        // every other leaf up to the first node already walked (included, so that the arms join).
        std::vector<std::vector<int>> arms(const std::vector<SheetSkeleton::Node> &nodes)
        {
            std::vector<char> has_child(nodes.size(), 0);
            for (const SheetSkeleton::Node &n : nodes)
                if (n.parent >= 0)
                    has_child[n.parent] = 1;
            std::vector<int> leaves;
            for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
                if (!has_child[i])
                    leaves.push_back(i);
            std::stable_sort(leaves.begin(), leaves.end(), [&](int a, int b)
                             { return nodes[a].depth > nodes[b].depth; });
            std::vector<char> walked(nodes.size(), 0);
            std::vector<std::vector<int>> out;
            for (int leaf : leaves)
            {
                std::vector<int> arm;
                int n = leaf;
                for (; n >= 0 && !walked[n]; n = nodes[n].parent)
                {
                    walked[n] = 1;
                    arm.push_back(n);
                }
                if (n >= 0)
                    arm.push_back(n);
                out.push_back(std::move(arm));
            }
            return out;
        }
    } // namespace

    std::vector<SheetTrace> trace_sheets(const SheetSkeleton &skeleton,
                                         const std::function<std::optional<int>(int)> &winding,
                                         const std::function<float(int)> &pull, const TraceParams &params)
    {
        const auto &nodes = skeleton.nodes();
        const int every = std::max(params.every_px, 1);
        std::vector<SheetTrace> traces;
        auto keep = [&](const std::vector<int> &arm, size_t begin, size_t end, int w)
        {
            if (static_cast<int>(end - begin) < params.min_length_px)
                return;
            SheetTrace trace;
            trace.winding = w;
            for (size_t i = begin; i < end; ++i)
            {
                trace.pull = std::max(trace.pull, pull(arm[i]));
                if ((i - begin) % every == 0 || i + 1 == end)
                    trace.points.emplace_back(nodes[arm[i]].x, nodes[arm[i]].y);
            }
            traces.push_back(std::move(trace));
        };
        for (const std::vector<int> &arm : arms(nodes))
        {
            size_t begin = 0;
            std::optional<int> current;
            for (size_t i = 0; i <= arm.size(); ++i)
            {
                const std::optional<int> w = i < arm.size() ? winding(arm[i]) : std::nullopt;
                if (i < arm.size() && w == current)
                    continue;
                if (current)
                    keep(arm, begin, i, *current);
                begin = i;
                current = w;
            }
        }
        return traces;
    }
} // namespace pw
