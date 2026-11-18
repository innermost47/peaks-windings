// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "layers.hpp"
#include "parallel.hpp"
#include "zarr_reader.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/QR>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <climits>
#include <limits>
#include <memory>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    using pw::Point;
    using Clock = std::chrono::steady_clock;

    const double NaN = std::numeric_limits<double>::quiet_NaN();

    double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

    // Python's a % b for b > 0 (never negative).
    int python_mod(int a, int b) { return ((a % b) + b) % b; }

    struct PatchPoint
    {
        double x = 0, y = 0, s = 0;
    };

    using Slices = std::map<int, std::map<int, PatchPoint>>;

    std::vector<int> largest_arc(std::vector<int> rays, int n_rays, int max_gap = 4)
    {
        std::sort(rays.begin(), rays.end());
        rays.erase(std::unique(rays.begin(), rays.end()), rays.end());
        const int n = static_cast<int>(rays.size());
        if (n == 0)
            return {};
        std::vector<int> gaps(n);
        for (int i = 0; i < n; ++i)
        {
            const int g = python_mod(rays[(i + 1) % n] - rays[i], n_rays);
            gaps[i] = g ? g : n_rays;
        }
        std::vector<int> breaks;
        for (int i = 0; i < n; ++i)
            if (gaps[i] > max_gap)
                breaks.push_back(i);
        if (breaks.empty()) // the layer goes all the way round: cut it at its widest gap
            breaks.push_back(static_cast<int>(std::max_element(gaps.begin(), gaps.end()) - gaps.begin()));
        std::vector<int> best;
        for (size_t k = 0; k < breaks.size(); ++k)
        {
            const int i = breaks[k], j = breaks[(k + 1) % breaks.size()];
            const int start = (i + 1) % n;
            const int wrapped = python_mod(j - i, n);
            const int length = wrapped ? wrapped : n;
            if (length > static_cast<int>(best.size()))
            {
                best.clear();
                for (int t = 0; t < length; ++t)
                    best.push_back(rays[(start + t) % n]);
            }
        }
        return best;
    }

    // The densest layer (or the one asked), from the counts of the survey.
    int choose_layer(const pw::LayerSurvey &survey, bool has_target, int target_layer)
    {
        if (survey.layer_order.empty())
            throw std::runtime_error("the result files hold no point");
        int chosen = survey.layer_order[0];
        if (has_target)
        {
            if (!survey.layer_points.count(target_layer))
                throw std::runtime_error("layer " + std::to_string(target_layer) + " has no point");
            chosen = target_layer;
        }
        else
        {
            for (int w : survey.layer_order)
                if (survey.layer_points.at(w) > survey.layer_points.at(chosen))
                    chosen = w;
        }
        std::printf("--> Analyzing layer: Winding #%d (%zu total points)\n", chosen, survey.layer_points.at(chosen));
        return chosen;
    }

    // The summits that see the layer, the one with the most points first: (track, number of points).
    std::vector<std::pair<int, size_t>> tracks_of_layer(const pw::LayerSurvey &survey, int layer)
    {
        const auto found = survey.track_points.find(layer);
        if (found == survey.track_points.end())
            return {};
        std::vector<std::pair<int, size_t>> tracks(found->second.begin(), found->second.end());
        std::stable_sort(tracks.begin(), tracks.end(), [](const auto &a, const auto &b)
                         { return a.second > b.second; });
        return tracks;
    }

    // The largest continuous angular arc of the layer around one summit. Rays are renumbered along the arc so that
    // it never jumps across angle 0.
    Slices extract_patch(const std::vector<Point> &points, int n_rays, int layer, int track)
    {
        std::vector<size_t> selected;
        std::vector<int> rays;
        for (size_t i = 0; i < points.size(); ++i)
            if (points[i].winding == layer && points[i].track == track)
            {
                selected.push_back(i);
                rays.push_back(points[i].ray);
            }
        Slices slices;
        const std::vector<int> arc = largest_arc(rays, n_rays);
        if (arc.empty())
            return slices;
        const int first = arc[0];
        std::map<int, int> position;
        for (int r : arc)
            position[r] = python_mod(r - first, n_rays);
        size_t kept = 0;
        for (size_t i : selected)
        {
            const auto it = position.find(points[i].ray);
            if (it == position.end())
                continue;
            ++kept;
            slices[points[i].z][it->second] = {points[i].x, points[i].y, 0.0};
        }
        const int span = python_mod(arc.back() - first, n_rays);
        std::printf("--> Summit (track) %d: rays %d to %d (arc of %.0f degrees, %zu points, %zu slices)\n", track, first,
                    arc.back(), span * 360.0 / n_rays, kept, slices.size());
        return slices;
    }

    void solve_patch(Slices &slices, double prior_sigma = 1e3)
    {
        std::vector<PatchPoint *> nodes;
        std::vector<double> guess;
        std::map<std::pair<int, int>, int> index;
        for (auto &[z, rays] : slices)
            for (auto &[ray, p] : rays)
            {
                index[{z, ray}] = static_cast<int>(nodes.size());
                nodes.push_back(&p);
                guess.push_back(static_cast<double>(ray));
            }
        const int n = static_cast<int>(nodes.size());
        if (n == 0)
            throw std::runtime_error("the patch is empty");

        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
        auto relation = [&](int a, int b, double target, double sigma) { // s(b) - s(a) = target
            const double w2 = 1.0 / (sigma * sigma);
            triplets.emplace_back(a, a, w2);
            triplets.emplace_back(b, b, w2);
            triplets.emplace_back(a, b, -w2);
            triplets.emplace_back(b, a, -w2);
            g[b] += w2 * target;
            g[a] -= w2 * target;
        };
        for (const auto &[z, rays] : slices)
            for (auto a = rays.begin(); a != rays.end(); ++a)
            {
                const auto b = std::next(a);
                if (b == rays.end())
                    break;
                if (b->first - a->first > 3)
                    continue;
                const double d = std::hypot(b->second.x - a->second.x, b->second.y - a->second.y);
                relation(index[{z, a->first}], index[{z, b->first}], d, 0.3);
            }
        for (auto a = slices.begin(); a != slices.end(); ++a)
        {
            const auto b = std::next(a);
            if (b == slices.end())
                break;
            for (const auto &ray : a->second)
                if (b->second.count(ray.first))
                    relation(index[{a->first, ray.first}], index[{b->first, ray.first}], 0.0, 1.5);
        }
        const double prior = 1.0 / (prior_sigma * prior_sigma);
        for (int i = 0; i < n; ++i)
            triplets.emplace_back(i, i, prior);
        const double measured_sigma = 1e-4; // the first node is held at 0
        triplets.emplace_back(0, 0, 1.0 / (measured_sigma * measured_sigma));

        Eigen::SparseMatrix<double> normal(n, n);
        normal.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(normal);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("the flat coordinates could not be solved");
        // the pull toward the guesses is moved onto the current solution at every pass, so it fades out
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(guess.data(), n);
        for (int pass = 0; pass < 6; ++pass)
        {
            const Eigen::VectorXd x_new = solver.solve(g + prior * x);
            const double step = (x_new - x).cwiseAbs().maxCoeff();
            x = x_new;
            if (step < 1e-12)
                break;
        }
        for (int i = 0; i < n; ++i)
            nodes[i]->s = x[i];
    }

    struct Guide
    {
        std::vector<double> s, x, y;
        std::vector<char> jump; // jump[i]: the points i and i + 1 are too far apart to be joined

        void mark_jumps(double max_jump)
        {
            jump.assign(s.size(), 0);
            if (max_jump <= 0 || s.size() < 3)
                return;
            std::vector<double> steps;
            for (size_t i = 0; i + 1 < s.size(); ++i)
                steps.push_back(std::hypot(x[i + 1] - x[i], y[i + 1] - y[i]));
            std::vector<double> sorted(steps);
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double limit = max_jump * sorted[sorted.size() / 2];
            for (size_t i = 0; i < steps.size(); ++i)
                jump[i] = steps[i] > limit;
        }

        void at(double q, double &xq, double &yq) const
        {
            if (!(q >= s.front() && q <= s.back()))
            {
                xq = yq = NaN;
                return;
            }
            size_t hi = std::lower_bound(s.begin(), s.end(), q) - s.begin();
            hi = std::min(std::max<size_t>(hi, 1), s.size() - 1);
            const size_t lo = hi - 1;
            if (!jump.empty() && jump[lo])
            {
                xq = yq = NaN;
                return;
            }
            const double t = q - s[lo], span = s[hi] - s[lo];
            xq = (x[hi] - x[lo]) / span * t + x[lo];
            yq = (y[hi] - y[lo]) / span * t + y[lo];
        }
    };

    struct Rail
    {
        std::vector<double> xs, ys, nx, ny; // the sheet in one slice and its unit normal, at every flat position
    };

    template <class T>
    void sample_rail(const cv::Mat &patch, const Rail &r, int half_thick, int x_min, int y_min, int z, double *row,
                     cv::Vec3f *where, int k0, int k1)
    {
        const int h = patch.rows, w = patch.cols;
        auto voxel = [&](int y, int x)
        { return static_cast<float>(patch.at<T>(y, x)); };
        for (int k = k0; k < k1; ++k)
        {
            if (!std::isfinite(r.xs[k]) || !std::isfinite(r.ys[k]) || !std::isfinite(r.nx[k]) || !std::isfinite(r.ny[k]))
                continue;
            double best = 0;
            for (int off = -half_thick; off <= half_thick; ++off)
            {
                // bilinear sample at the point moved by `off` voxels across the sheet
                const double px = std::min(std::max(r.xs[k] + off * r.nx[k] - x_min, 0.0), w - 1.001);
                const double py = std::min(std::max(r.ys[k] + off * r.ny[k] - y_min, 0.0), h - 1.001);
                const int x0 = static_cast<int>(px), y0 = static_cast<int>(py);
                const double ax = px - x0, ay = py - y0;
                const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                const double v = voxel(y0, x0) * (1 - ax) * (1 - ay) + voxel(y0, x1) * ax * (1 - ay) +
                                 voxel(y1, x0) * (1 - ax) * ay + voxel(y1, x1) * ax * ay;
                best = off == -half_thick ? v : std::max(best, v);
            }
            row[k] = std::isfinite(best) ? best : 0.0;
            where[k] = cv::Vec3f(static_cast<float>(r.xs[k]), static_cast<float>(r.ys[k]), static_cast<float>(z));
        }
    }

    // numpy.gradient of a 1-D array with unit spacing.
    std::vector<double> gradient(const std::vector<double> &v)
    {
        const size_t n = v.size();
        std::vector<double> out(n, 0.0);
        if (n < 2)
            return out;
        out[0] = v[1] - v[0];
        out[n - 1] = v[n - 1] - v[n - 2];
        for (size_t i = 1; i + 1 < n; ++i)
            out[i] = (v[i + 1] - v[i - 1]) / 2.0;
        return out;
    }

    // numpy.percentile (linear interpolation) of an unsorted array, which is reordered.
    double percentile(std::vector<double> &v, double q)
    {
        const double pos = (v.size() - 1) * q / 100.0;
        const size_t lo = static_cast<size_t>(std::floor(pos));
        std::nth_element(v.begin(), v.begin() + lo, v.end());
        const double a = v[lo];
        const double b = lo + 1 < v.size() ? *std::min_element(v.begin() + lo + 1, v.end()) : a;
        const double t = pos - lo;
        return t >= 0.5 ? b - (b - a) * (1.0 - t) : a + (b - a) * t;
    }

    // A float32 array of shape (rows, cols, 3) as a NumPy .npy file.
    void save_npy(const std::string &path, const cv::Mat &data)
    {
        CV_Assert(data.type() == CV_32FC3 && data.isContinuous());
        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(data.rows) + ", " +
                             std::to_string(data.cols) + ", 3), }";
        while ((10 + header.size() + 1) % 64 != 0)
            header += ' ';
        header += '\n';
        std::ofstream f(path, std::ios::binary);
        const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                         static_cast<unsigned char>(header.size() & 0xff),
                                         static_cast<unsigned char>(header.size() >> 8)};
        f.write(reinterpret_cast<const char *>(magic), 10);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        f.write(reinterpret_cast<const char *>(data.data), static_cast<std::streamsize>(data.total() * 3 * sizeof(float)));
        if (!f)
            throw std::runtime_error("cannot write " + path);
    }

    // A "nice" tick step (1, 2 or 5 times a power of ten) giving about `target` ticks over `span`.
    double tick_step(double span, int target)
    {
        const double raw = std::max(span, 1e-9) / target;
        const double power = std::pow(10.0, std::floor(std::log10(raw)));
        for (double m : {1.0, 2.0, 5.0, 10.0})
            if (raw <= m * power)
                return m * power;
        return 10.0 * power;
    }

    // The image with a title and graduated axes (position along the sheet, slice Z).
    void save_figure(const std::string &path, const cv::Mat &gray, const std::string &title, int z_start, int z_end,
                     const std::string &x_label, double x_max)
    {
        const int W = 2400, H = 1200, left = 190, right = 50, top = 90, bottom = 120;
        const cv::Scalar black(0, 0, 0);
        cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        const cv::Rect plot(left, top, W - left - right, H - top - bottom);
        cv::Mat scaled, colour;
        const bool shrink = gray.cols >= plot.width && gray.rows >= plot.height;
        cv::resize(gray, scaled, plot.size(), 0, 0, shrink ? cv::INTER_AREA : cv::INTER_LINEAR);
        cv::cvtColor(scaled, colour, cv::COLOR_GRAY2BGR);
        colour.copyTo(canvas(plot));
        cv::rectangle(canvas, plot, black, 2);
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        auto centred = [&](const std::string &text, int cx, int baseline, double scale, int thickness)
        {
            int base = 0;
            const cv::Size size = cv::getTextSize(text, font, scale, thickness, &base);
            cv::putText(canvas, text, cv::Point(cx - size.width / 2, baseline), font, scale, black, thickness, cv::LINE_AA);
        };
        centred(title, W / 2, 55, 1.0, 2);
        const double x_step = tick_step(x_max, 8);
        for (double v = 0; v <= x_max + 1e-9; v += x_step)
        {
            const int x = plot.x + static_cast<int>(std::lround(v / x_max * plot.width));
            cv::line(canvas, cv::Point(x, plot.br().y), cv::Point(x, plot.br().y + 10), black, 2);
            centred(std::to_string(static_cast<long long>(v)), x, plot.br().y + 42, 0.8, 1);
        }
        centred(x_label, plot.x + plot.width / 2, H - 25, 0.9, 2);
        const double z_lo = z_start - 0.5, z_hi = z_end + 0.5;
        const double z_step = std::max(1.0, tick_step(z_hi - z_lo, 8));
        for (double v = std::ceil(z_start / z_step) * z_step; v <= z_end; v += z_step)
        {
            const int y = plot.y + static_cast<int>(std::lround((v - z_lo) / (z_hi - z_lo) * plot.height));
            cv::line(canvas, cv::Point(plot.x - 10, y), cv::Point(plot.x, y), black, 2);
            const std::string text = std::to_string(static_cast<long long>(v));
            int base = 0;
            const cv::Size size = cv::getTextSize(text, font, 0.8, 1, &base);
            cv::putText(canvas, text, cv::Point(plot.x - 18 - size.width, y + size.height / 2), font, 0.8, black, 1,
                        cv::LINE_AA);
        }
        { // the vertical axis label, written flat then turned
            const std::string text = "Slice Z (native voxels)";
            int base = 0;
            const cv::Size size = cv::getTextSize(text, font, 0.9, 2, &base);
            cv::Mat label(size.height + base + 8, size.width + 8, CV_8UC3, cv::Scalar(255, 255, 255));
            cv::putText(label, text, cv::Point(4, size.height + 4), font, 0.9, black, 2, cv::LINE_AA);
            cv::rotate(label, label, cv::ROTATE_90_COUNTERCLOCKWISE);
            const cv::Rect where(12, plot.y + (plot.height - label.rows) / 2, label.cols, label.rows);
            if ((where & cv::Rect(0, 0, W, H)) == where)
                label.copyTo(canvas(where));
        }
        if (!cv::imwrite(path, canvas))
            throw std::runtime_error("cannot write " + path);
    }

    struct Options
    {
        std::string scroll_name = "scroll"; // folders: results/{json,png,csv}/<scroll name>/
        std::string results_dir;            // empty: results/json/<scroll name>
        bool has_layer = false;
        int layer = 0;
        std::string scroll;
        bool all = true; // all the summits together (false: one image per summit)
        int slab = 6;
        int step_z = 1;
        std::string output; // empty: results/png/<scroll name>/layer_<N>/unrolled_layer_<N>_<all|clean>.png
        int threads = 0;
        int connections = 32;
        double block_mb = 256;
        int z_min = INT_MIN, z_max = INT_MAX;
        int tracks = 5;         // summits unrolled (those with the most points of the layer)
        bool has_track = false; // or one given summit
        int track = 0;
        int max_width = 1600;           // widest image along the sheet (px)
        double max_jump = 5.0;          // see Guide::mark_jumps (0: every step is interpolated)
        double max_pull = 3.0;          // points the network pulled further than this (sigmas) are left out (0: none is)
        double fill_deg = 10.0;         // holes along the angle up to this wide are filled (degrees)
        int fill_z = 25;                // holes along Z up to this many slices are filled
        int smooth_z = 9;               // running median of the radius along Z, over this many slices (1: none)
        int smooth_deg_cols = 3;        // running mean of the radius along the angle, over this many columns (1: none)
        int centre_window = 101;        // running median of the centre along Z, over this many slices
        bool centre_from_files = false; // the centre of the result files instead of the circle fitted to the layer
        int sectors = 16;               // the turn is read in this many sectors (each its own region of the volume)
    };

    // The rails of one image: the sheet in every sampled slice, and what the image is.
    struct Unrolling
    {
        std::vector<int> dense_z;      // the slices sampled (rows)
        int width = 0;                 // columns
        std::function<Rail(int)> rail; // the sheet in slice z, one point per column (NaN where unknown)
        std::string title, x_label;
        double x_max = 0; // what the right edge of the image stands for (x_label's unit)
    };

    // One summit: guide rails interpolated between the slices of its patch.
    bool rails_of_patch(const Slices &slices, const Options &o, Unrolling &u)
    {
        double s_min = std::numeric_limits<double>::infinity(), s_max = -s_min;
        for (const auto &slice : slices)
            for (const auto &ray : slice.second)
            {
                s_min = std::min(s_min, ray.second.s);
                s_max = std::max(s_max, ray.second.s);
            }
        const int width = std::max(300, std::min(o.max_width, static_cast<int>(std::nearbyint(s_max - s_min))));
        auto s_grid = std::make_shared<std::vector<double>>(width);
        const double s_step = (s_max - s_min) / (width - 1);
        for (int i = 0; i < width; ++i)
            (*s_grid)[i] = i * s_step + s_min;
        (*s_grid)[width - 1] = s_max;

        const int z_start = slices.begin()->first, z_end = slices.rbegin()->first;
        u.dense_z.clear();
        for (int z = z_start; z <= z_end; z += o.step_z)
            u.dense_z.push_back(z);
        u.width = width;
        u.x_label = "Position along sheet (voxels)";
        u.x_max = s_max - s_min;

        std::printf("--> Unrolling:\n");
        std::printf("    - Width: %d pixels along sheet (%.2f voxels per pixel)\n", width, (s_max - s_min) / (width - 1));
        std::printf("    - Height: %zu consecutive CT slices (Z=%d to %d)\n", u.dense_z.size(), z_start, z_end);
        std::printf("    - Thickness: %d voxels in Maximum Intensity Projection (MIP)\n", o.slab);

        // one guide per slice of the patch (slices with fewer than two distinct positions are left out)
        std::vector<int> z_keys;
        std::map<int, Guide> guides;
        for (const auto &[z, rays] : slices)
        {
            std::vector<const PatchPoint *> pts;
            for (const auto &ray : rays)
                pts.push_back(&ray.second);
            std::stable_sort(pts.begin(), pts.end(), [](const PatchPoint *a, const PatchPoint *b)
                             { return a->s < b->s; });
            Guide guide;
            for (const PatchPoint *p : pts)
                if (guide.s.empty() || p->s != guide.s.back())
                {
                    guide.s.push_back(p->s);
                    guide.x.push_back(p->x);
                    guide.y.push_back(p->y);
                }
            if (guide.s.size() < 2)
                continue;
            guide.mark_jumps(o.max_jump);
            z_keys.push_back(z);
            guides[z] = std::move(guide);
        }
        if (z_keys.empty())
        {
            std::printf("    no slice of the patch has two points: nothing to unroll\n");
            return false;
        }
        auto keys = std::make_shared<std::vector<int>>(std::move(z_keys));
        auto shared_guides = std::make_shared<std::map<int, Guide>>(std::move(guides));
        u.rail = [keys, shared_guides, s_grid, width](int z)
        {
            const std::vector<int> &z_keys = *keys;
            const std::map<int, Guide> &guides = *shared_guides;
            const int i = static_cast<int>(std::upper_bound(z_keys.begin(), z_keys.end(), z) - z_keys.begin()) - 1;
            const int k_prev = z_keys[std::max(i, 0)];
            const int k_next = z > k_prev ? z_keys[std::min<int>(i + 1, static_cast<int>(z_keys.size()) - 1)] : k_prev;
            const double weight = k_prev == k_next ? 0.0 : (z - k_prev) / static_cast<double>(k_next - k_prev);
            const Guide &a = guides.at(k_prev);
            const Guide &b = guides.at(k_next);
            Rail r;
            r.xs.resize(width);
            r.ys.resize(width);
            for (int k = 0; k < width; ++k)
            {
                double xa, ya, xb, yb;
                a.at((*s_grid)[k], xa, ya);
                b.at((*s_grid)[k], xb, yb);
                r.xs[k] = (1.0 - weight) * xa + weight * xb;
                r.ys[k] = (1.0 - weight) * ya + weight * yb;
            }
            const std::vector<double> dx = gradient(r.xs), dy = gradient(r.ys);
            r.nx.resize(width);
            r.ny.resize(width);
            for (int k = 0; k < width; ++k)
            {
                const double norm = std::hypot(dx[k], dy[k]) + 1e-6;
                r.nx[k] = -dy[k] / norm;
                r.ny[k] = dx[k] / norm;
            }
            return r;
        };

        return true;
    }

    // Samples the CT slices along the rails (slab MIP), sector by sector, and writes the image, its figure and the
    // coordinates of its pixels.
    void sample_and_save(const Unrolling &u, const Options &o, pw::ZarrVolume *volume, const std::string &output_png)
    {
        const int threads = o.threads > 0 ? o.threads : pw::hardware_threads();
        const std::vector<int> &dense_z = u.dense_z;
        const int width = u.width;
        const auto t0 = Clock::now();
        const int half_thick = o.slab / 2;
        const auto &rail = u.rail;
        if (dense_z.empty())
            return;

        // the columns of every sector
        const int n_sectors = std::max(1, std::min(o.sectors, width));
        std::vector<std::pair<int, int>> sectors;
        for (int s = 0; s < n_sectors; ++s)
            sectors.emplace_back(s * width / n_sectors, (s + 1) * width / n_sectors);

        // extent of every rail in every sector (the rails themselves are computed again when they are sampled)
        struct Extent
        {
            bool seen = false;
            double x_min = 0, x_max = 0, y_min = 0, y_max = 0;
        };
        std::vector<std::vector<Extent>> extents(dense_z.size(), std::vector<Extent>(sectors.size()));
        pw::parallel_for(static_cast<int>(dense_z.size()), threads, [&](int i, int)
                         {
        const Rail r = rail(dense_z[i]);
        for (size_t s = 0; s < sectors.size(); ++s) {
            Extent& e = extents[i][s];
            for (int k = sectors[s].first; k < sectors[s].second; ++k) {
                if (!std::isfinite(r.xs[k]) || !std::isfinite(r.ys[k])) continue;
                if (!e.seen) {
                    e.seen = true;
                    e.x_min = e.x_max = r.xs[k];
                    e.y_min = e.y_max = r.ys[k];
                }
                e.x_min = std::min(e.x_min, r.xs[k]), e.x_max = std::max(e.x_max, r.xs[k]);
                e.y_min = std::min(e.y_min, r.ys[k]), e.y_max = std::max(e.y_max, r.ys[k]);
            }
        } });
        auto merged = [&](size_t first, size_t last, size_t s) { // extent of the rails first..last in sector s
            Extent m;
            for (size_t i = first; i <= last; ++i)
            {
                const Extent &e = extents[i][s];
                if (!e.seen)
                    continue;
                if (!m.seen)
                    m = e;
                m.x_min = std::min(m.x_min, e.x_min), m.x_max = std::max(m.x_max, e.x_max);
                m.y_min = std::min(m.y_min, e.y_min), m.y_max = std::max(m.y_max, e.y_max);
            }
            return m;
        };
        // whether the region covering the rails first..last of sector s fits in the memory budget (2 bytes per voxel)
        auto block_fits = [&](size_t first, size_t last, size_t s)
        {
            const Extent m = merged(first, last, s);
            if (!m.seen)
                return true;
            const double w = m.x_max - m.x_min + 2 * half_thick + 10;
            const double h = m.y_max - m.y_min + 2 * half_thick + 10;
            return (last - first + 1) * w * h * 2 <= o.block_mb * 1048576.0;
        };

        // blocks: for every sector, runs of slices inside one storage chunk along Z, within the memory budget
        const int depth = static_cast<int>(std::max<int64_t>(1, volume->chunks()[0]));
        struct Block
        {
            size_t first, last, sector;
        };
        std::vector<Block> blocks;
        for (size_t s = 0; s < sectors.size(); ++s)
        {
            size_t first = 0;
            for (size_t i = 1; i <= dense_z.size(); ++i)
                if (i == dense_z.size() || dense_z[i] / depth != dense_z[first] / depth || !block_fits(first, i, s))
                {
                    blocks.push_back({first, i - 1, s});
                    first = i;
                }
        }
        // in Z order, the sectors of one run together (what the volume's cache has just decoded is read again)
        std::stable_sort(blocks.begin(), blocks.end(), [](const Block &a, const Block &b)
                         { return a.first < b.first; });
        std::printf("    - Read in %zu block(s) (chunk depth %d, %zu sector(s))\n", blocks.size(), depth, sectors.size());
        std::fflush(stdout);

        cv::Mat image = cv::Mat::zeros(static_cast<int>(dense_z.size()), width, CV_64F);
        // where every pixel was taken in the volume: (x, y, z), NaN where nothing was sampled
        cv::Mat coords(image.size(), CV_32FC3, cv::Scalar::all(std::numeric_limits<float>::quiet_NaN()));
        size_t done = 0, next_report = 0;
        for (const Block &bl : blocks)
        {
            const size_t b0 = bl.first, b1 = bl.last;
            const Extent m = merged(b0, b1, bl.sector);
            const int k0 = sectors[bl.sector].first, k1 = sectors[bl.sector].second;
            if (m.seen)
            {
                const int x_min = std::max(0, static_cast<int>(m.x_min - half_thick - 5));
                const int x_max = static_cast<int>(m.x_max + half_thick + 5);
                const int y_min = std::max(0, static_cast<int>(m.y_min - half_thick - 5));
                const int y_max = static_cast<int>(m.y_max + half_thick + 5);
                const std::vector<cv::Mat> block =
                    volume->read_region(dense_z[b0], dense_z[b1] + 1, y_min, y_max, x_min, x_max);
                pw::parallel_for(static_cast<int>(b1 - b0 + 1), threads, [&](int i, int)
                                 {
                const int z = dense_z[b0 + i];
                const size_t plane = static_cast<size_t>(z - dense_z[b0]);
                if (plane >= block.size() || block[plane].empty()) return;
                // the plane is read in its own type (u8, u16...): no float copy of the whole region per thread
                const cv::Mat& patch = block[plane];
                if (patch.channels() != 1) throw std::runtime_error("the volume must have a single channel");
                const Rail r = rail(z);
                double* row = image.ptr<double>(static_cast<int>(b0) + i);
                cv::Vec3f* where = coords.ptr<cv::Vec3f>(static_cast<int>(b0) + i);
                switch (patch.depth()) {
                    case CV_8U: sample_rail<unsigned char>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_8S: sample_rail<signed char>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_16U: sample_rail<unsigned short>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_16S: sample_rail<short>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_32S: sample_rail<int>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_32F: sample_rail<float>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    case CV_64F: sample_rail<double>(patch, r, half_thick, x_min, y_min, z, row, where, k0, k1); break;
                    default: throw std::runtime_error("unsupported voxel type");
                } });
            }
            if (bl.sector + 1 == sectors.size())
                done = b1 + 1;
            if (done >= next_report || &bl == &blocks.back())
            {
                std::printf("    Sampled slices: %zu/%zu (%.0f s)\n", done, dense_z.size(), seconds_since(t0));
                std::fflush(stdout);
                next_report = done + std::max<size_t>(1, dense_z.size() / 20);
            }
        }
        // contrast: the 3 % .. 97 % range of the sampled values, the unsampled pixels staying black
        std::vector<double> seen;
        for (int r = 0; r < image.rows; ++r)
        {
            const double *row = image.ptr<double>(r);
            for (int c = 0; c < image.cols; ++c)
                if (row[c] > 0)
                    seen.push_back(row[c]);
        }
        if (!seen.empty())
        {
            const double p3 = percentile(seen, 3.0), p97 = percentile(seen, 97.0);
            if (p97 > p3)
                for (int r = 0; r < image.rows; ++r)
                {
                    double *row = image.ptr<double>(r);
                    for (int c = 0; c < image.cols; ++c)
                        row[c] = row[c] > 0 ? std::min(std::max((row[c] - p3) / (p97 - p3), 0.0), 1.0) : 0.0;
                }
        }
        // 8-bit grey as matplotlib's "gray" colormap gives it
        unsigned char lut[256];
        for (int i = 0; i < 256; ++i)
            lut[i] = static_cast<unsigned char>((i == 255 ? 1.0 : i * (1.0 / 255.0)) * 255.0);
        cv::Mat gray(image.size(), CV_8U);
        for (int r = 0; r < image.rows; ++r)
        {
            const double *row = image.ptr<double>(r);
            unsigned char *out = gray.ptr<unsigned char>(r);
            for (int c = 0; c < image.cols; ++c)
                out[c] = lut[std::min(std::max(static_cast<int>(row[c] * 256.0), 0), 255)];
        }

        const std::filesystem::path out_path(output_png);
        if (!out_path.parent_path().empty())
            std::filesystem::create_directories(out_path.parent_path());
        const std::string full_png =
            (out_path.parent_path() / (out_path.stem().string() + "_full.png")).string();
        if (!cv::imwrite(full_png, gray))
            throw std::runtime_error("cannot write " + full_png);
        const std::string coords_npy = (out_path.parent_path() / (out_path.stem().string() + "_coords.npy")).string();
        save_npy(coords_npy, coords);
        char range[64];
        const int z_start = dense_z.front(), z_end = dense_z.back();
        std::snprintf(range, sizeof(range), " | Z = [%d -> %d]", z_start, z_end);
        save_figure(output_png, gray, u.title + range, z_start, z_end, u.x_label, u.x_max);

        std::printf("\n--> Unrolling done in %.1f s.\n", seconds_since(t0));
        std::printf("--> Image saved to: '%s' (full resolution: '%s', coordinates: '%s')\n\n", output_png.c_str(),
                    full_png.c_str(), coords_npy.c_str());
    }

    // ------------------------------------------------------------------------------------------- all summits together
    const double TWO_PI = 6.283185307179586476925286766559;

    // Running median of the finite values of v over `window` neighbours (odd), NaN kept where v is NaN.
    std::vector<double> running_median(const std::vector<double> &v, int window)
    {
        const int n = static_cast<int>(v.size()), half = std::max(0, window / 2);
        std::vector<double> out(v), buf;
        if (half == 0)
            return out;
        for (int i = 0; i < n; ++i)
        {
            if (!std::isfinite(v[i]))
                continue;
            buf.clear();
            for (int j = std::max(0, i - half); j <= std::min(n - 1, i + half); ++j)
                if (std::isfinite(v[j]))
                    buf.push_back(v[j]);
            std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
            out[i] = buf[buf.size() / 2];
        }
        return out;
    }

    // Fills the runs of NaN of v no longer than max_gap by straight lines between their two ends (on a circle when
    // circular: the last value is next to the first).
    void fill_gaps(std::vector<double> &v, int max_gap, bool circular)
    {
        const int n = static_cast<int>(v.size());
        std::vector<int> known;
        for (int i = 0; i < n; ++i)
            if (std::isfinite(v[i]))
                known.push_back(i);
        if (known.size() < 2 || max_gap <= 0)
            return;
        const size_t pairs = circular ? known.size() : known.size() - 1;
        for (size_t k = 0; k < pairs; ++k)
        {
            const int a = known[k], b = known[(k + 1) % known.size()];
            const int gap = (b - a + n) % n - 1; // the NaN between a and b
            if (gap <= 0 || gap > max_gap || (!circular && b < a))
                continue;
            for (int t = 1; t <= gap; ++t)
            {
                const double w = t / static_cast<double>(gap + 1);
                v[(a + t) % n] = (1.0 - w) * v[a] + w * v[b];
            }
        }
    }

    // The layer around the centre of every slice: its radius for every slice and every angle.
    struct Ring
    {
        int z_first = 0, z_last = -1;
        int width = 0;                   // column k: the angle (k + 0.5) * 360 / width degrees
        std::vector<double> r;           // (z - z_first) * width + k; NaN where the sheet is not known
        std::vector<double> cx, cy;      // the centre of every slice (smoothed)
        size_t points = 0, left_out = 0; // points used, points left out (pulled too far)

        double &at(int z, int k) { return r[static_cast<size_t>(z - z_first) * width + k]; }
        double at(int z, int k) const { return r[static_cast<size_t>(z - z_first) * width + k]; }
    };

    // The centres of the slices, smoothed along Z: a running median over `window` slices, then a running mean over a
    // fifth of it (what is left of the jitter), the nearest one where a slice has none.
    void smooth_centres(const std::map<int, pw::Centre> &given, int z_first, int z_last, int window, Ring &ring)
    {
        const int nz = z_last - z_first + 1;
        ring.cx.assign(nz, NaN);
        ring.cy.assign(nz, NaN);
        std::vector<int> zs;
        std::vector<double> xs, ys;
        for (const auto &[z, c] : given)
        {
            zs.push_back(z);
            xs.push_back(c.x);
            ys.push_back(c.y);
        }
        if (zs.empty())
            return;
        xs = running_median(xs, window);
        ys = running_median(ys, window);
        const int half = std::max(1, window / 10);
        std::vector<double> mx(xs.size()), my(ys.size());
        for (size_t i = 0; i < xs.size(); ++i)
        {
            double sx = 0, sy = 0;
            int n = 0;
            for (size_t j = i >= static_cast<size_t>(half) ? i - half : 0; j <= std::min(xs.size() - 1, i + half); ++j)
                sx += xs[j], sy += ys[j], ++n;
            mx[i] = sx / n;
            my[i] = sy / n;
        }
        xs = mx;
        ys = my;
        for (int i = 0; i < nz; ++i)
        { // the nearest given slice
            const int z = z_first + i;
            const size_t hi = std::lower_bound(zs.begin(), zs.end(), z) - zs.begin();
            size_t best = hi < zs.size() ? hi : zs.size() - 1;
            if (hi > 0 && (hi == zs.size() || z - zs[hi - 1] <= zs[hi] - z))
                best = hi - 1;
            ring.cx[i] = xs[best];
            ring.cy[i] = ys[best];
        }
    }

    Ring build_ring(const pw::LayerSurvey &survey, int layer, const Options &o)
    {
        Ring ring;
        ring.width = o.max_width;
        const int width = ring.width;
        // the slices where the layer is seen, and their centres. By default the circle that best fits the layer's own
        // points in the slice (least squares, Kasa): it follows the layer and moves smoothly with it. The centre the
        // result files give is one of the summits, and jumps from one to another as they relay each other: the angle
        // around it would jump with it, and the image would be torn there (--centre files to use it anyway).
        std::map<int, pw::Centre> centres;
        int z_first = INT_MAX, z_last = INT_MIN;
        struct Sums
        {
            double n = 0, x = 0, y = 0, xx = 0, yy = 0, xy = 0, xz = 0, yz = 0, z = 0; // z here: x^2 + y^2
        };
        std::map<int, Sums> sums;
        double ox = NaN, oy = NaN; // an origin near the points, for the sums to stay well conditioned
        pw::for_each_layer_slice(o.results_dir, o.z_min, o.z_max, o.threads, survey, layer, {},
                                 [&](int z, const std::vector<Point> &pts)
                                 {
                                     z_first = std::min(z_first, z);
                                     z_last = std::max(z_last, z);
                                     Sums &m = sums[z];
                                     for (const Point &p : pts)
                                     {
                                         if (o.max_pull > 0 && std::fabs(p.pull) > o.max_pull)
                                             continue;
                                         if (!std::isfinite(ox))
                                             ox = p.x, oy = p.y;
                                         const double x = p.x - ox, y = p.y - oy, q = x * x + y * y;
                                         m.n += 1, m.x += x, m.y += y, m.xx += x * x, m.yy += y * y, m.xy += x * y;
                                         m.xz += x * q, m.yz += y * q, m.z += q;
                                     }
                                 });
        if (z_first > z_last)
            return ring;
        ring.z_first = z_first;
        ring.z_last = z_last;
        if (o.centre_from_files && !survey.centres.empty())
        {
            for (const auto &[z, c] : survey.centres)
                if (z >= z_first - o.centre_window && z <= z_last + o.centre_window)
                    centres[z] = c;
            std::printf("--> Centre: the one of every result file, smoothed over %d slices\n", o.centre_window);
        }
        else
        {
            // x^2 + y^2 + D x + E y + F = 0 in the least squares sense: the centre is (-D/2, -E/2)
            for (const auto &[z, m] : sums)
            {
                if (m.n < 10)
                    continue;
                Eigen::Matrix3d A;
                A << m.xx, m.xy, m.x, m.xy, m.yy, m.y, m.x, m.y, m.n;
                const Eigen::Vector3d b(-m.xz, -m.yz, -m.z);
                const Eigen::Vector3d sol = A.colPivHouseholderQr().solve(b);
                if (!sol.allFinite())
                    continue;
                centres[z] = {ox - sol[0] / 2, oy - sol[1] / 2};
            }
            std::printf("--> Centre: the circle fitted to the layer in every slice, smoothed over %d slices\n",
                        o.centre_window);
        }
        if (centres.empty())
            throw std::runtime_error("no slice has enough points of the layer to place its centre");
        smooth_centres(centres, z_first, z_last, o.centre_window, ring);

        // the radius of the sheet at every angle: the median of the points of all summits in that angle
        const int nz = z_last - z_first + 1;
        ring.r.assign(static_cast<size_t>(nz) * width, NaN);
        pw::for_each_layer_slice(o.results_dir, o.z_min, o.z_max, o.threads, survey, layer, {},
                                 [&](int z, const std::vector<Point> &pts)
                                 {
                                     const int i = z - z_first;
                                     const double cx = ring.cx[i], cy = ring.cy[i];
                                     std::vector<std::pair<int, double>> cells;
                                     for (const Point &p : pts)
                                     {
                                         if (o.max_pull > 0 && std::fabs(p.pull) > o.max_pull)
                                         {
                                             ++ring.left_out;
                                             continue;
                                         }
                                         double a = std::atan2(p.y - cy, p.x - cx);
                                         if (a < 0)
                                             a += TWO_PI;
                                         const int k = std::min(width - 1, static_cast<int>(a / TWO_PI * width));
                                         cells.emplace_back(k, std::hypot(p.x - cx, p.y - cy));
                                         ++ring.points;
                                     }
                                     std::sort(cells.begin(), cells.end());
                                     for (size_t a = 0; a < cells.size();)
                                     {
                                         size_t b = a;
                                         while (b < cells.size() && cells[b].first == cells[a].first)
                                             ++b;
                                         const size_t m = a + (b - a) / 2; // cells are sorted by radius within a column
                                         ring.at(z, cells[a].first) = (b - a) % 2 ? cells[m].second
                                                                                  : 0.5 * (cells[m - 1].second + cells[m].second);
                                         a = b;
                                     }
                                 });
        // holes along the angle, then along Z; then the radius smoothed along Z and along the angle
        const int fill_cols = static_cast<int>(std::lround(o.fill_deg / 360.0 * width));
        size_t known_before = 0, known_after = 0;
        for (double v : ring.r)
            known_before += std::isfinite(v);
        std::vector<double> row(width), col(nz);
        for (int i = 0; i < nz; ++i)
        {
            for (int k = 0; k < width; ++k)
                row[k] = ring.r[static_cast<size_t>(i) * width + k];
            fill_gaps(row, fill_cols, true);
            for (int k = 0; k < width; ++k)
                ring.r[static_cast<size_t>(i) * width + k] = row[k];
        }
        for (int k = 0; k < width; ++k)
        {
            for (int i = 0; i < nz; ++i)
                col[i] = ring.r[static_cast<size_t>(i) * width + k];
            fill_gaps(col, o.fill_z, false);
            col = running_median(col, o.smooth_z);
            for (int i = 0; i < nz; ++i)
                ring.r[static_cast<size_t>(i) * width + k] = col[i];
        }
        if (o.smooth_deg_cols > 1)
            for (int i = 0; i < nz; ++i)
            {
                for (int k = 0; k < width; ++k)
                    row[k] = ring.r[static_cast<size_t>(i) * width + k];
                const int half = o.smooth_deg_cols / 2;
                for (int k = 0; k < width; ++k)
                {
                    if (!std::isfinite(row[k]))
                        continue;
                    double sum = 0;
                    int n = 0;
                    for (int d = -half; d <= half; ++d)
                    {
                        const double v = row[(k + d + width) % width];
                        if (std::isfinite(v) && std::fabs(v - row[k]) < 0.5 * std::fabs(row[k]))
                            sum += v, ++n; // not across the seam where the layer meets the next turn
                    }
                    ring.r[static_cast<size_t>(i) * width + k] = sum / n;
                }
            }
        for (double v : ring.r)
            known_after += std::isfinite(v);
        std::printf("--> Layer #%d with all its summits: %zu points (%zu pulled further than %.1f sigma left out), "
                    "slices %d to %d\n",
                    layer, ring.points, ring.left_out, o.max_pull, z_first, z_last);
        std::printf("    - Known: %.1f %% of the image from the points, %.1f %% after filling the holes\n",
                    100.0 * known_before / ring.r.size(), 100.0 * known_after / ring.r.size());
        return ring;
    }

    // The rails of the ring: in slice z, column k is the point at the radius of the sheet and the angle of the column.
    void rails_of_ring(std::shared_ptr<const Ring> ring, const Options &o, Unrolling &u)
    {
        u.dense_z.clear();
        for (int z = ring->z_first; z <= ring->z_last; z += o.step_z)
            u.dense_z.push_back(z);
        u.width = ring->width;
        u.x_label = "Angle around the centre (degrees)";
        u.x_max = 360.0;
        const int width = ring->width;
        u.rail = [ring, width](int z)
        {
            Rail r;
            r.xs.assign(width, NaN);
            r.ys.assign(width, NaN);
            const int i = z - ring->z_first;
            if (i >= 0 && z <= ring->z_last)
                for (int k = 0; k < width; ++k)
                {
                    const double rad = ring->at(z, k);
                    if (!std::isfinite(rad))
                        continue;
                    const double a = (k + 0.5) / width * TWO_PI;
                    r.xs[k] = ring->cx[i] + rad * std::cos(a);
                    r.ys[k] = ring->cy[i] + rad * std::sin(a);
                }
            const std::vector<double> dx = gradient(r.xs), dy = gradient(r.ys);
            r.nx.resize(width);
            r.ny.resize(width);
            for (int k = 0; k < width; ++k)
            {
                const double norm = std::hypot(dx[k], dy[k]) + 1e-6;
                r.nx[k] = -dy[k] / norm;
                r.ny[k] = dx[k] / norm;
            }
            return r;
        };
        std::printf("--> Unrolling:\n");
        std::printf("    - Width: %d columns, 0 to 360 degrees around the centre (%.3f degree per column)\n", width,
                    360.0 / width);
        std::printf("    - Height: %zu CT slices (Z=%d to %d)\n", u.dense_z.size(), ring->z_first, ring->z_last);
        std::printf("    - Thickness: %d voxels in Maximum Intensity Projection (MIP)\n", o.slab);
    }

    void usage()
    {
        std::cout << "Unroll one winding layer of a scanned scroll into a flat image (slab MIP).\n\n"
                     "  --scroll-name NAME  name of the scroll, the same as given to peaks_windings (default: scroll);\n"
                     "                      results are read from results/json/NAME/ and written to\n"
                     "                      results/png/NAME/layer_<N>/ and results/csv/NAME/layer_<N>/\n"
                     "  --results-dir DIR   folder containing the JSON results (default: results/json/NAME)\n"
                     "  --layer N           winding layer to unroll (default: the one with the most points)\n"
                     "  --scroll URL        CT volume to sample: path or URL of a zarr store (required)\n"
                     "  --summits all|each  all the summits together, one image of the layer around the centre (default),\n"
                     "                      or one image per summit (the former unrolling)\n"
                     "  --slab N            slab thickness in voxels for the MIP (default: 6)\n"
                     "  --step-z N          vertical step in voxels (default: 1)\n"
                     "  --output FILE       output image (default: results/png/NAME/layer_<N>/unrolled_layer_<N>_all.png,\n"
                     "                      or ..._clean.png with --summits each);\n"
                     "                      _track<N> is added to the name for every summit\n"
                     "  --tracks N          each: the N summits with the most points of the layer (default: 5)\n"
                     "  --track N           each: one given summit only\n"
                     "  --max-pull F        all: points pulled further than F sigma by the network are left out\n"
                     "                      (default: 3; 0: none)\n"
                     "  --fill-deg F        all: holes along the angle up to F degrees are filled (default: 10)\n"
                     "  --fill-z N          all: holes along Z up to N slices are filled (default: 25)\n"
                     "  --smooth-z N        all: running median of the sheet along Z over N slices (default: 9; 1: none)\n"
                     "  --smooth-cols N     all: running mean of the sheet along the angle over N columns (default: 3)\n"
                     "  --centre fit|files  all: the circle fitted to the layer in every slice (default), or the centre of\n"
                     "                      the result files (one of the summits: it jumps when they relay each other)\n"
                     "  --centre-window N   all: running median of the centre along Z over N slices (default: 101)\n"
                     "  --sectors N         the turn is read in N sectors of the volume (default: 16)\n"
                     "  --max-width N       widest image along the sheet in px; all: the number of angles (default: 1600)\n"
                     "  --max-jump F        a step longer than F times the median step of its slice is left blank\n"
                     "                      (default: 5; 0: every step is interpolated)\n"
                     "  --block-mb F        memory budget of one read block in MB (default: 256)\n"
                     "  --z-min N           first slice kept (default: all)\n"
                     "  --z-max N           last slice kept (default: all)\n"
                     "  --threads N         worker threads (default 0: all); --workers is the same option\n"
                     "  --connections N     parallel http downloads (default 32)\n";
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                usage();
                return 0;
            }
            std::string value;
            const size_t eq = arg.find('=');
            if (eq != std::string::npos)
            {
                value = arg.substr(eq + 1);
                arg = arg.substr(0, eq);
            }
            else
            {
                if (i + 1 >= argc)
                    throw std::invalid_argument("option " + arg + " needs a value");
                value = argv[++i];
            }
            if (arg == "--scroll-name")
                o.scroll_name = value;
            else if (arg == "--results-dir")
                o.results_dir = value;
            else if (arg == "--layer")
                o.layer = std::stoi(value), o.has_layer = true;
            else if (arg == "--scroll")
                o.scroll = value;
            else if (arg == "--slab")
                o.slab = std::stoi(value);
            else if (arg == "--step-z")
                o.step_z = std::stoi(value);
            else if (arg == "--output")
                o.output = value;
            else if (arg == "--block-mb")
                o.block_mb = std::stod(value);
            else if (arg == "--z-min")
                o.z_min = std::stoi(value);
            else if (arg == "--z-max")
                o.z_max = std::stoi(value);
            else if (arg == "--threads" || arg == "--workers")
                o.threads = std::stoi(value);
            else if (arg == "--connections")
                o.connections = std::stoi(value);
            else if (arg == "--tracks")
                o.tracks = std::stoi(value);
            else if (arg == "--track")
                o.track = std::stoi(value), o.has_track = true;
            else if (arg == "--max-width")
                o.max_width = std::max(300, std::stoi(value));
            else if (arg == "--max-jump")
                o.max_jump = std::stod(value);
            else if (arg == "--summits")
            {
                if (value != "all" && value != "each")
                    throw std::invalid_argument("--summits must be all or each");
                o.all = value == "all";
            }
            else if (arg == "--max-pull")
                o.max_pull = std::stod(value);
            else if (arg == "--fill-deg")
                o.fill_deg = std::stod(value);
            else if (arg == "--fill-z")
                o.fill_z = std::stoi(value);
            else if (arg == "--smooth-z")
                o.smooth_z = std::max(1, std::stoi(value));
            else if (arg == "--smooth-cols")
                o.smooth_deg_cols = std::max(1, std::stoi(value));
            else if (arg == "--centre-window")
                o.centre_window = std::max(1, std::stoi(value));
            else if (arg == "--centre")
            {
                if (value != "fit" && value != "files")
                    throw std::invalid_argument("--centre must be fit or files");
                o.centre_from_files = value == "files";
            }
            else if (arg == "--sectors")
                o.sectors = std::max(1, std::stoi(value));
            else
                throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
        if (o.step_z < 1)
            throw std::invalid_argument("--step-z must be at least 1");
        if (o.slab < 0)
            throw std::invalid_argument("--slab cannot be negative");
        const bool is_number = !o.scroll.empty() && o.scroll.find_first_not_of("0123456789") == std::string::npos;
        if (o.scroll.empty() || is_number)
            throw std::invalid_argument(
                "--scroll must be the path or URL of the zarr CT volume, not a number");
        if (o.scroll_name.empty())
            throw std::invalid_argument("--scroll-name cannot be empty");
        if (o.results_dir.empty())
            o.results_dir = (std::filesystem::path("results") / "json" / o.scroll_name).string();

        // 1. The layer, and the summits that see it
        // Two passes over the files, so that a whole scroll fits in memory: the first aligns the labels and counts
        // the points (two slices held at a time), the second keeps only the points of the chosen layer and summits.
        const pw::LayerSurvey survey = pw::survey_layers(o.results_dir, o.z_min, o.z_max, o.threads);
        const int n_rays = pw::ray_count(o.results_dir);
        const int layer = choose_layer(survey, o.has_layer, o.layer);
        // every output of this layer goes to its own folders: results/png/<scroll>/layer_<N>/, results/csv/<scroll>/layer_<N>/
        const std::string layer_folder = "layer_" + std::to_string(layer);
        const std::filesystem::path png_dir = std::filesystem::path("results") / "png" / o.scroll_name / layer_folder;
        const std::filesystem::path csv_dir = std::filesystem::path("results") / "csv" / o.scroll_name / layer_folder;
        const std::filesystem::path base =
            !o.output.empty() ? std::filesystem::path(o.output)
                              : png_dir / ("unrolled_layer_" + std::to_string(layer) + "_clean.png");

        if (o.all)
        {
            // 2. The layer around the centre, every summit together; 3. sampling
            const auto ranked = tracks_of_layer(survey, layer);
            std::printf("--> %zu summit(s) see this layer: all of them together\n", ranked.size());
            auto ring = std::make_shared<const Ring>(build_ring(survey, layer, o));
            if (ring->z_first > ring->z_last)
            {
                std::printf("--> Layer %d has no point\n", layer);
                return 0;
            }
            {
                // the smoothed centre of every slice, to check whether the scroll axis is tilted
                std::filesystem::create_directories(csv_dir);
                const std::filesystem::path csv_path = csv_dir / "centres.csv";
                std::ofstream csv(csv_path);
                if (!csv)
                    throw std::runtime_error("cannot write " + csv_path.string());
                csv << "z,cx,cy\n";
                for (int i = 0; i < static_cast<int>(ring->cx.size()); ++i)
                    csv << ring->z_first + i << "," << ring->cx[i] << "," << ring->cy[i] << "\n";
                std::printf("--> Centres written to: '%s'\n", csv_path.string().c_str());
            }
            const std::unique_ptr<pw::ZarrVolume> volume = pw::ZarrVolume::open(o.scroll, o.connections, o.threads);
            std::printf("--> %s\n", volume->description().c_str());
            Unrolling u;
            rails_of_ring(ring, o, u);
            u.title = "Unrolled layer " + std::to_string(layer) + " from all summits (slab MIP)";
            const std::string out_png =
                o.output.empty() ? (png_dir / ("unrolled_layer_" + std::to_string(layer) + "_all.png")).string()
                                 : base.string();
            sample_and_save(u, o, volume.get(), out_png);
            return 0;
        }

        std::vector<int> tracks;
        if (o.has_track)
        {
            tracks.push_back(o.track);
        }
        else
        {
            const auto ranked = tracks_of_layer(survey, layer);
            std::printf("--> %zu summit(s) see this layer; unrolling the %zu with the most points\n", ranked.size(),
                        std::min<size_t>(ranked.size(), std::max(o.tracks, 0)));
            for (size_t i = 0; i < ranked.size() && static_cast<int>(i) < o.tracks; ++i)
                tracks.push_back(ranked[i].first);
        }
        const std::vector<Point> points =
            tracks.empty() ? std::vector<Point>()
                           : pw::load_layer(o.results_dir, o.z_min, o.z_max, o.threads, survey, layer, tracks);
        const std::unique_ptr<pw::ZarrVolume> volume = pw::ZarrVolume::open(o.scroll, o.connections, o.threads);
        std::printf("--> %s\n", volume->description().c_str());

        for (int track : tracks)
        {
            // 2. Largest continuous patch around this summit, 3. flat coordinates, 4. sampling
            Slices slices = extract_patch(points, n_rays, layer, track);
            if (slices.empty())
            {
                std::printf("--> Summit (track) %d has no point of layer %d\n", track, layer);
                continue;
            }
            solve_patch(slices);
            const std::string out_png =
                (base.parent_path() / (base.stem().string() + "_track" + std::to_string(track) + ".png")).string();
            Unrolling u;
            if (!rails_of_patch(slices, o, u))
                continue;
            u.title = "Unrolled layer " + std::to_string(layer) + " from summit " + std::to_string(track) + " (slab MIP)";
            sample_and_save(u, o, volume.get(), out_png);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}