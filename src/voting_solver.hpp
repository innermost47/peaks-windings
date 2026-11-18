// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pw
{
    // Labels related by observations "label[b] = apply(label[a], value)", in a group:
    //   IntegerShift  labels are integers, value the difference           (the windings of the sheets)
    //   SignFlip      labels are +-1, value +1 (same) or -1 (opposite)     (the orientation of the sheets)
    struct IntegerShift
    {
        static int identity() { return 0; }
        static int apply(int label, int value) { return label + value; }
        static int invert(int value) { return -value; }
        static int distance(int a, int b) { return std::abs(a - b); }
    };

    struct SignFlip
    {
        static int identity() { return 1; }
        static int apply(int label, int value) { return label * value; }
        static int invert(int value) { return value; }
        static int distance(int a, int b) { return a == b ? 0 : 1; }
    };

    // One observation: label[b] = apply(label[a], value), or label[a] = value when b < 0 (an anchor).
    struct Observation
    {
        int a = 0, b = -1;
        int value = 0;
        double weight = 1;
    };

    // Labelling from observations that mostly agree, as in quality-guided phase unwrapping.
    //
    // The observations between two unknowns are first pooled: each possible relation gets the weight of the
    // observations that support it, and the pair keeps the best supported one, with the margin over the second best
    // as its confidence. The labels then grow from the anchors along the most confident pairs first (a maximum
    // spanning tree), so that an inconsistency (a sheet missed along a ray, a fold, two sheets touching) ends up on
    // the least supported links instead of being smeared over all of them. Finally every unknown takes, in turn, the
    // label most supported by all its neighbours, until nothing changes.
    template <class Group>
    class VotingSolver
    {
    public:
        struct Params
        {
            int refine_passes = 20; // at most this many passes of neighbour voting
        };

        struct Result
        {
            std::vector<int> label;    // per unknown; meaningless where !reached
            std::vector<char> reached; // whether the unknown is linked to an anchor
            std::vector<int> residual; // per observation: how far it is from being met (0: met)
            double tension = 0;        // weight of the observations not met / weight of all of them
        };

        explicit VotingSolver(const Params &params) : params_(params) {}
        VotingSolver() : VotingSolver(Params{}) {}

        Result solve(int n, const std::vector<Observation> &observations) const
        {
            Result result;
            result.label.assign(n + 1, Group::identity());
            result.reached.assign(n + 1, 0);
            const int ground = n; // stands for the anchors: label[a] = value is label[a] = apply(ground, value)
            const std::vector<Pair> pairs = pool(n, observations);
            std::vector<std::vector<int>> pairs_of(n + 1);
            for (int i = 0; i < static_cast<int>(pairs.size()); ++i)
            {
                pairs_of[pairs[i].lo].push_back(i);
                pairs_of[pairs[i].hi].push_back(i);
            }
            grow(pairs, pairs_of, ground, result);
            refine(pairs, pairs_of, n, result);

            result.residual.resize(observations.size());
            double total = 0, unmet = 0;
            for (size_t i = 0; i < observations.size(); ++i)
            {
                const Observation &o = observations[i];
                const int from = o.b < 0 ? Group::identity() : result.label[o.a];
                const int to = o.b < 0 ? result.label[o.a] : result.label[o.b];
                result.residual[i] = Group::distance(Group::apply(from, o.value), to);
                total += o.weight;
                if (result.residual[i] != 0)
                    unmet += o.weight;
            }
            result.tension = total > 0 ? unmet / total : 0;
            result.label.pop_back();
            result.reached.pop_back();
            return result;
        }

    private:
        // All the observations between two unknowns lo < hi, as votes on the relation label[hi] = apply(label[lo], v).
        struct Pair
        {
            int lo = 0, hi = 0;
            std::vector<std::pair<int, double>> votes; // (value, weight)
            int best = 0;
            double confidence = 0; // weight of the best value minus that of the second best

            void add(int value, double weight)
            {
                for (auto &v : votes)
                    if (v.first == value)
                    {
                        v.second += weight;
                        return;
                    }
                votes.emplace_back(value, weight);
            }

            void settle()
            {
                std::sort(votes.begin(), votes.end(), [](const auto &a, const auto &b)
                          { return a.second > b.second; });
                best = votes.front().first;
                confidence = votes.front().second - (votes.size() > 1 ? votes[1].second : 0.0);
            }

            int other(int node) const { return node == lo ? hi : lo; }
            // The label of `node` that the value v gives, from the label of the other end.
            int label_from(int node, int other_label, int v) const
            {
                return node == hi ? Group::apply(other_label, v) : Group::apply(other_label, Group::invert(v));
            }
        };

        static std::vector<Pair> pool(int n, const std::vector<Observation> &observations)
        {
            std::unordered_map<std::uint64_t, int> index;
            index.reserve(observations.size());
            std::vector<Pair> pairs;
            for (const Observation &o : observations)
            {
                // an anchor label[a] = value is label[a] = apply(label[ground], value): ground -> a
                const int from = o.b < 0 ? n : o.a, to = o.b < 0 ? o.a : o.b;
                const int lo = std::min(from, to), hi = std::max(from, to);
                const int value = from == lo ? o.value : Group::invert(o.value);
                const std::uint64_t key = static_cast<std::uint64_t>(lo) << 32 | static_cast<std::uint32_t>(hi);
                const auto [it, fresh] = index.emplace(key, static_cast<int>(pairs.size()));
                if (fresh)
                    pairs.push_back({lo, hi, {}, 0, 0});
                pairs[it->second].add(value, o.weight);
            }
            for (Pair &p : pairs)
                p.settle();
            return pairs;
        }

        // Growth from the ground along the most confident pairs first.
        static void grow(const std::vector<Pair> &pairs, const std::vector<std::vector<int>> &pairs_of, int ground,
                         Result &result)
        {
            using Entry = std::pair<double, int>; // (confidence, pair)
            std::priority_queue<Entry> queue;
            auto reach = [&](int node, int label)
            {
                result.reached[node] = 1;
                result.label[node] = label;
                for (int p : pairs_of[node])
                    if (!result.reached[pairs[p].other(node)])
                        queue.emplace(pairs[p].confidence, p);
            };
            reach(ground, Group::identity());
            while (!queue.empty())
            {
                const Pair &p = pairs[queue.top().second];
                queue.pop();
                const bool lo_in = result.reached[p.lo], hi_in = result.reached[p.hi];
                if (lo_in == hi_in)
                    continue;
                const int from = lo_in ? p.lo : p.hi, to = p.other(from);
                reach(to, p.label_from(to, result.label[from], p.best));
            }
        }

        // Every unknown takes the label most supported by all its neighbours.
        void refine(const std::vector<Pair> &pairs, const std::vector<std::vector<int>> &pairs_of, int n,
                    Result &result) const
        {
            std::vector<std::pair<int, double>> candidates;
            for (int pass = 0; pass < params_.refine_passes; ++pass)
            {
                bool changed = false;
                for (int node = 0; node < n; ++node)
                {
                    if (!result.reached[node])
                        continue;
                    candidates.clear();
                    for (int pi : pairs_of[node])
                    {
                        const Pair &p = pairs[pi];
                        for (const auto &[v, w] : p.votes)
                            candidates.emplace_back(p.label_from(node, result.label[p.other(node)], v), w);
                    }
                    std::sort(candidates.begin(), candidates.end());
                    int best = result.label[node];
                    double best_w = -1, current_w = 0;
                    for (size_t i = 0; i < candidates.size();)
                    {
                        size_t j = i;
                        double w = 0;
                        for (; j < candidates.size() && candidates[j].first == candidates[i].first; ++j)
                            w += candidates[j].second;
                        if (candidates[i].first == result.label[node])
                            current_w = w;
                        if (w > best_w)
                            best_w = w, best = candidates[i].first;
                        i = j;
                    }
                    if (best != result.label[node] && best_w > current_w)
                    {
                        result.label[node] = best;
                        changed = true;
                    }
                }
                if (!changed)
                    break;
            }
        }

        Params params_;
    };
} // namespace pw
