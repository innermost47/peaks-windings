// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "scroll_mask.hpp"

#include "numeric.hpp"
#include "windings.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

namespace pw
{
    namespace
    {
        // An elliptic kernel of px_full full-resolution pixels, at the working scale (odd size, at least 3).
        cv::Mat kernel(double px_full, double scale)
        {
            const int k = std::max(3, round_half_even(px_full * scale) | 1);
            return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        }

        // Interior of the scroll and the spires without the frame (low resolution).
        void interior_and_spires(const cv::Mat &img, double scale, double frame_px, double close_px, cv::Mat &mask,
                                 cv::Mat &spires, double &threshold)
        {
            cv::Mat thresh_raw;
            threshold = cv::threshold(img, thresh_raw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            cv::Mat frame, frame_zone;
            cv::morphologyEx(thresh_raw, frame, cv::MORPH_OPEN, kernel(frame_px, scale));
            cv::dilate(frame, frame_zone, kernel(frame_px * 11.0 / 9.0, scale));
            cv::Mat spires_only = cv::Mat::zeros(img.size(), CV_8U);
            thresh_raw.copyTo(spires_only, frame_zone == 0);
            cv::Mat grouped, pad;
            cv::morphologyEx(spires_only, grouped, cv::MORPH_CLOSE, kernel(close_px, scale));
            cv::copyMakeBorder(grouped, pad, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::Mat flood = pad.clone();
            cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
            const cv::Rect inner(1, 1, img.cols, img.rows);
            cv::Mat shape_mask = (pad(inner) == 255) | (flood(inner) == 0);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(shape_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            mask = cv::Mat::zeros(img.size(), CV_8U);
            if (!contours.empty())
            {
                const auto largest = std::max_element(contours.begin(), contours.end(),
                                                      [](const auto &a, const auto &b)
                                                      { return cv::contourArea(a) < cv::contourArea(b); });
                cv::drawContours(mask, contours, static_cast<int>(largest - contours.begin()), cv::Scalar(255),
                                 cv::FILLED);
            }
            spires = cv::Mat::zeros(img.size(), CV_8U);
            spires_only.copyTo(spires, mask);
        }
    } // namespace

    ScrollMasks detect_scroll_masks(const cv::Mat &img_u8, const Params &params)
    {
        ScrollMasks m;
        cv::Mat small;
        cv::resize(img_u8, small, cv::Size(), params.scale, params.scale, cv::INTER_AREA);
        double threshold = 0;
        interior_and_spires(small, params.scale, params.frame_px, params.close_px, m.mask_small, m.spires_small,
                            threshold);
        // full resolution: inside the upsampled interior, above the same threshold, specks removed
        cv::Mat mask;
        cv::resize(m.mask_small, mask, img_u8.size(), 0, 0, cv::INTER_NEAREST);
        m.inside = (mask > 0) / 255;
        cv::Mat above = (img_u8 > threshold) / 255;
        cv::Mat raw;
        cv::bitwise_and(above, m.inside, raw);
        cv::morphologyEx(raw, m.spires, cv::MORPH_OPEN, cv::Mat::ones(2, 2, CV_8U));
        return m;
    }
} // namespace pw
