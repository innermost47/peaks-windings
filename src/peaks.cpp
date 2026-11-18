// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "peaks.hpp"

#include <algorithm>
#include <cmath>

namespace pw
{
    std::vector<Peak> select_peaks(const std::vector<Peak> &candidates, int n, TrackState *state, double radius,
                                   int memory)
    {
        std::vector<Peak> chosen;
        if (!state)
        {
            for (int i = 0; i < static_cast<int>(candidates.size()) && i < n; ++i)
            {
                chosen.push_back(candidates[i]);
                chosen.back().track = i;
            }
            return chosen;
        }
        struct Pair
        {
            double distance;
            int track, candidate;
        };
        std::vector<Pair> pairs;
        for (int t = 0; t < static_cast<int>(state->tracks.size()); ++t)
            for (int c = 0; c < static_cast<int>(candidates.size()); ++c)
            {
                const double d = std::hypot(candidates[c].x - state->tracks[t].x, candidates[c].y - state->tracks[t].y);
                if (d < radius)
                    pairs.push_back({d, t, c});
            }
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b)
                         { return a.distance < b.distance; });
        std::vector<int> track_of(candidates.size(), -1), candidate_of(state->tracks.size(), -1);
        for (const Pair &pair : pairs)
            if (track_of[pair.candidate] < 0 && candidate_of[pair.track] < 0)
            {
                track_of[pair.candidate] = pair.track;
                candidate_of[pair.track] = pair.candidate;
            }
        std::vector<int> taken; // candidate indices
        for (int c = 0; c < static_cast<int>(candidates.size()) && static_cast<int>(taken.size()) < n; ++c)
            if (track_of[c] >= 0)
                taken.push_back(c);
        for (int c = 0; c < static_cast<int>(candidates.size()) && static_cast<int>(taken.size()) < n; ++c)
            if (track_of[c] < 0)
                taken.push_back(c);
        std::sort(taken.begin(), taken.end()); // back to height order

        std::vector<char> seen(state->tracks.size(), 0);
        std::vector<TrackState::Track> born;
        for (int c : taken)
        {
            Peak peak = candidates[c];
            if (track_of[c] >= 0)
            {
                TrackState::Track &track = state->tracks[track_of[c]];
                track.x = peak.x, track.y = peak.y, track.missed = 0;
                seen[track_of[c]] = 1;
                peak.track = track.id;
            }
            else
            {
                peak.track = state->next_id++;
                born.push_back({peak.track, peak.x, peak.y, 0});
            }
            chosen.push_back(peak);
        }
        std::vector<TrackState::Track> alive;
        for (size_t t = 0; t < state->tracks.size(); ++t)
        {
            TrackState::Track track = state->tracks[t];
            if (!seen[t] && ++track.missed > memory)
                continue; // the track ends
            alive.push_back(track);
        }
        alive.insert(alive.end(), born.begin(), born.end());
        state->tracks = std::move(alive);
        return chosen;
    }
} // namespace pw
