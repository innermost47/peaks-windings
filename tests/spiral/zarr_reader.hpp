// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
namespace pw
{
    struct Spiral
    {
        static constexpr double R0 = 40, PITCH = 30;
        static double cx(double z) { return 1500 + 0.3 * z; }
        static double cy(double z) { return 1500 - 0.2 * z; }
        // distance (px) from (x, y) to the nearest turn of the sheet in slice z, and that turn's angle
        static double distance(double x, double y, double z, double &theta)
        {
            const double dx = x - cx(z), dy = y - cy(z);
            theta = std::atan2(dy, dx);
            if (theta < 0)
                theta += 2 * M_PI;
            const double u = (std::hypot(dx, dy) - R0) / PITCH - theta / (2 * M_PI);
            return std::fabs(u - std::round(u)) * PITCH;
        }
    };
    class ZarrVolume
    {
    public:
        static std::unique_ptr<ZarrVolume> open(const std::string &, int, int, double = 2.0)
        {
            return std::make_unique<ZarrVolume>();
        }
        std::string description() const { return "synthetic spiral scroll"; }
        std::array<long long, 3> shape() const { return {400, 3000, 3000}; }
        std::array<long long, 3> chunks() const { return {128, 128, 128}; }
        std::vector<cv::Mat> read_region(int z0, int z1, int y0, int y1, int x0, int x1) const
        {
            std::vector<cv::Mat> out;
            for (int z = z0; z < z1; ++z)
            {
                cv::Mat m(y1 - y0, x1 - x0, CV_8U);
                for (int y = 0; y < m.rows; ++y)
                    for (int x = 0; x < m.cols; ++x)
                    {
                        double th;
                        const double d = Spiral::distance(x0 + x, y0 + y, z, th);
                        // the sheet: bright, with a texture of fibres that goes on from slice to slice
                        const double fibre = 0.5 + 0.5 * std::sin(th * 40 + z * 0.05) * std::cos(z * 0.02 + th * 7);
                        m.at<unsigned char>(y, x) = d < 2.0 ? (unsigned char)(120 + 120 * fibre) : 20;
                    }
                out.push_back(m);
            }
            return out;
        }
    };
}
