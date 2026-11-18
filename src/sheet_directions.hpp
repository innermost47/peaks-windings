// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace pw
{
    // The direction of the sheets everywhere in a slice, measured at the working scale: the structure tensor of the
    // spires, smoothed over `sigma_px` full-resolution pixels so that it bridges the breaks of the sheets. A direction
    // has no sign (a sheet runs both ways): it is kept as twice its angle.
    class SheetDirections
    {
    public:
        SheetDirections(const cv::Mat &spires_small, const cv::Mat &mask_small, double scale, double sigma_px);

        // The direction of the sheets at a full-resolution point (unit vector, of either sign).
        cv::Point2d tangent(double x, double y) const;

        // The number of turns the direction of the sheets makes around a working-scale point on a closed loop of
        // offsets (the Poincare index: +1 around nested sheets, +1/2 at the tip of a fold); false when the loop
        // leaves the scroll.
        bool index(int x, int y, const std::vector<cv::Point> &loop, double &turns) const;

        // A circle of `radius` working-scale pixels, as offsets for index().
        static std::vector<cv::Point> circle(double radius);

        double scale() const { return scale_; }
        cv::Size size() const { return doubled_angle_.size(); }

    private:
        bool inside(const cv::Point &p) const;

        double scale_;
        cv::Mat mask_;
        cv::Mat doubled_angle_; // CV_32F: twice the angle of the normal to the sheets, in [0, 2 pi)
    };
} // namespace pw
