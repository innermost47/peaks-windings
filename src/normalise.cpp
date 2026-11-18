// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "windings.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pw
{
    cv::Mat normalise(const cv::Mat &slice)
    {
        CV_Assert(slice.channels() == 1);
        const size_t total = slice.total();
        cv::Mat out(slice.size(), CV_8U);
        if (total == 0)
            return out;
        // numpy.percentile (linear interpolation) at 1 % and 99 %
        const double pos[2] = {(total - 1) * 0.01, (total - 1) * 0.99};
        float pct[2] = {0, 0};
        auto stretch = [&](float v)
        {
            const float span = std::max(pct[1] - pct[0], 1e-4f);
            const float s = (v - pct[0]) / span * 255.0f;
            return static_cast<unsigned char>(std::min(std::max(s, 0.0f), 255.0f));
        };
        if (slice.depth() == CV_8U || slice.depth() == CV_16U)
        {
            const int bins = slice.depth() == CV_8U ? 256 : 65536;
            std::vector<size_t> hist(bins, 0);
            for (int r = 0; r < slice.rows; ++r)
            {
                if (slice.depth() == CV_8U)
                {
                    const unsigned char *p = slice.ptr<unsigned char>(r);
                    for (int c = 0; c < slice.cols; ++c)
                        ++hist[p[c]];
                }
                else
                {
                    const unsigned short *p = slice.ptr<unsigned short>(r);
                    for (int c = 0; c < slice.cols; ++c)
                        ++hist[p[c]];
                }
            }
            auto order_statistic = [&](size_t rank)
            {
                size_t seen = 0;
                for (int v = 0; v < bins; ++v)
                {
                    seen += hist[v];
                    if (seen > rank)
                        return static_cast<double>(v);
                }
                return static_cast<double>(bins - 1);
            };
            for (int i = 0; i < 2; ++i)
            {
                const size_t lo = static_cast<size_t>(std::floor(pos[i]));
                const size_t hi = std::min(lo + 1, total - 1);
                const double a = order_statistic(lo), b = order_statistic(hi);
                pct[i] = static_cast<float>(a + (b - a) * (pos[i] - lo));
            }
            std::vector<unsigned char> lut(bins);
            for (int v = 0; v < bins; ++v)
                lut[v] = stretch(static_cast<float>(v));
            for (int r = 0; r < slice.rows; ++r)
            {
                unsigned char *o = out.ptr<unsigned char>(r);
                if (slice.depth() == CV_8U)
                {
                    const unsigned char *p = slice.ptr<unsigned char>(r);
                    for (int c = 0; c < slice.cols; ++c)
                        o[c] = lut[p[c]];
                }
                else
                {
                    const unsigned short *p = slice.ptr<unsigned short>(r);
                    for (int c = 0; c < slice.cols; ++c)
                        o[c] = lut[p[c]];
                }
            }
            return out;
        }
        cv::Mat as_float;
        slice.convertTo(as_float, CV_32F);
        cv::Mat flat = as_float.clone().reshape(1, 1);
        float *begin = flat.ptr<float>(0);
        float *end = begin + total;
        for (int i = 0; i < 2; ++i)
        {
            const size_t lo = static_cast<size_t>(std::floor(pos[i]));
            std::nth_element(begin, begin + lo, end);
            const double a = begin[lo];
            const double b = lo + 1 < total ? *std::min_element(begin + lo + 1, end) : a;
            pct[i] = static_cast<float>(a + (b - a) * (pos[i] - lo));
        }
        for (int r = 0; r < slice.rows; ++r)
        {
            const float *p = as_float.ptr<float>(r);
            unsigned char *o = out.ptr<unsigned char>(r);
            for (int c = 0; c < slice.cols; ++c)
                o[c] = stretch(p[c]);
        }
        return out;
    }
} // namespace pw
