// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "windings.hpp"

#include <cstdio>

namespace pw
{
    namespace
    {
        // A number rounded to `decimals`, written as Python's repr() does (at least one digit after the point).
        void put_number(std::string &s, double v, int decimals)
        {
            char buf[64];
            int n = std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
            while (n > 2 && buf[n - 1] == '0' && buf[n - 2] != '.')
                --n;
            s.append(buf, n);
        }
    } // namespace

    std::string to_json(const SliceResult &result, long long z)
    {
        std::string s;
        size_t crossings = 0;
        for (const LineOut &line : result.lines)
            crossings += line.crossings.size();
        s.reserve(256 + result.lines.size() * 64 + crossings * 112);
        const std::string z_text = z >= 0 ? std::to_string(z) : "null";
        s += "{";
        if (z >= 0)
            s += "\"z\":" + z_text + ",";
        s += "\"n_rays\":" + std::to_string(result.n_rays) + ",";
        s += "\"lines\":[";
        bool first_line = true;
        for (const LineOut &line : result.lines)
        {
            if (!first_line)
                s += ",";
            first_line = false;
            s += "{\"ray_index\":" + std::to_string(line.ray_index);
            s += ",\"peak_index\":" + std::to_string(line.peak_index);
            s += ",\"track\":" + std::to_string(line.track);
            s += ",\"angle\":";
            put_number(s, line.angle, 4);
            s += ",\"crossings\":[";
            bool first = true;
            for (const CrossingOut &c : line.crossings)
            {
                if (!first)
                    s += ",";
                first = false;
                s += "{\"x\":";
                put_number(s, c.x, 2);
                s += ",\"y\":";
                put_number(s, c.y, 2);
                s += ",\"z\":" + z_text;
                s += ",\"winding_label\":" + std::to_string(c.winding_label);
                s += ",\"u_continuous\":";
                put_number(s, c.u_continuous, 3);
                s += ",\"fragment\":" + std::to_string(c.fragment);
                s += ",\"pull\":";
                put_number(s, c.pull, 2);
                s += "}";
            }
            s += "]}";
        }
        s += "],\"centre\":";
        if (result.has_centre)
        {
            s += "{\"x\":";
            put_number(s, result.cx, 1);
            s += ",\"y\":";
            put_number(s, result.cy, 1);
            s += ",\"peak_index\":" + std::to_string(result.centre_peak);
            s += ",\"track\":" + std::to_string(result.centre_track);
            s += ",\"handedness\":" + std::to_string(result.handedness) + "}";
        }
        else
        {
            s += "null";
        }
        s += ",\"peaks\":[";
        for (size_t i = 0; i < result.peaks.size(); ++i)
        {
            if (i)
                s += ",";
            s += "{\"x\":";
            put_number(s, result.peaks[i].x, 1);
            s += ",\"y\":";
            put_number(s, result.peaks[i].y, 1);
            s += ",\"height\":";
            put_number(s, result.peaks[i].height, 2);
            s += ",\"score\":";
            put_number(s, result.scores[i], 3);
            s += ",\"track\":" + std::to_string(result.peaks[i].track);
            s += "}";
        }
        s += "]}";
        return s;
    }
} // namespace pw
