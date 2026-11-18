// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "sheet_directions.hpp"

#include "numeric.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace pw
{
    SheetDirections::SheetDirections(const cv::Mat &spires_small, const cv::Mat &mask_small, double scale,
                                     double sigma_px)
        : scale_(scale), mask_(mask_small)
    {
        cv::Mat f, gx, gy;
        spires_small.convertTo(f, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(f, f, cv::Size(), 1.0);
        cv::Sobel(f, gx, CV_32F, 1, 0);
        cv::Sobel(f, gy, CV_32F, 0, 1);
        cv::Mat jxx = gx.mul(gx), jyy = gy.mul(gy), jxy = gx.mul(gy);
        for (cv::Mat *j : {&jxx, &jyy, &jxy})
            cv::GaussianBlur(*j, *j, cv::Size(), sigma_px * scale);
        cv::phase(jxx - jyy, 2 * jxy, doubled_angle_);
    }

    cv::Point2d SheetDirections::tangent(double x, double y) const
    {
        const int sx = std::clamp(static_cast<int>(std::lround(x * scale_)), 0, doubled_angle_.cols - 1);
        const int sy = std::clamp(static_cast<int>(std::lround(y * scale_)), 0, doubled_angle_.rows - 1);
        const double normal = 0.5 * doubled_angle_.at<float>(sy, sx);
        return {-std::sin(normal), std::cos(normal)};
    }

    bool SheetDirections::inside(const cv::Point &p) const
    {
        return p.x >= 0 && p.y >= 0 && p.x < mask_.cols && p.y < mask_.rows && mask_.at<unsigned char>(p) > 0;
    }

    std::vector<cv::Point> SheetDirections::circle(double radius)
    {
        const int samples = std::max(16, static_cast<int>(std::ceil(TWO_PI * radius / 2)));
        std::vector<cv::Point> loop(samples);
        for (int k = 0; k < samples; ++k)
            loop[k] = cv::Point(static_cast<int>(std::lround(radius * std::cos(TWO_PI * k / samples))),
                                static_cast<int>(std::lround(radius * std::sin(TWO_PI * k / samples))));
        return loop;
    }

    bool SheetDirections::index(int x, int y, const std::vector<cv::Point> &loop, double &turns) const
    {
        const cv::Point last = cv::Point(x, y) + loop.back();
        if (!inside(last))
            return false;
        double turned = 0;
        float previous = doubled_angle_.at<float>(last);
        for (const cv::Point &d : loop)
        {
            const cv::Point p(x + d.x, y + d.y);
            if (!inside(p))
                return false;
            const float a = doubled_angle_.at<float>(p);
            turned += wrap_angle(a - previous);
            previous = a;
        }
        turns = turned / (2 * TWO_PI); // the angle was doubled
        return true;
    }
} // namespace pw
