// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "windings.hpp"

#include "parallel.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace pw
{
    namespace
    {

        const double PI = 3.14159265358979323846;

        using Clock = std::chrono::steady_clock;

        double seconds_since(Clock::time_point t0)
        {
            return std::chrono::duration<double>(Clock::now() - t0).count();
        }

        // Python's round(): half to even (the default floating-point rounding mode).
        int round_half_even(double v) { return static_cast<int>(std::nearbyint(v)); }

        // ------------------------------------------------------------------------------------------- 1. spires
        // An elliptic kernel of px_full full-resolution pixels, at the working scale (odd size, at least 3).
        cv::Mat kernel(double px_full, double scale)
        {
            const int k = std::max(3, round_half_even(px_full * scale) | 1);
            return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        }

        // Interior of the scroll and the spires without the frame (low resolution). Anything thicker than frame_px
        // full-resolution pixels is the frame (the case).
        void detect_scroll_interior_mask(const cv::Mat &img, double scale, double frame_px, double close_px, cv::Mat &mask,
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
                int best = 0;
                double best_area = -1.0;
                for (int i = 0; i < static_cast<int>(contours.size()); ++i)
                {
                    const double area = cv::contourArea(contours[i]);
                    if (area > best_area)
                    {
                        best_area = area;
                        best = i;
                    }
                }
                cv::drawContours(mask, contours, best, cv::Scalar(255), cv::FILLED);
            }
            spires = cv::Mat::zeros(img.size(), CV_8U);
            spires_only.copyTo(spires, mask);
        }

        // Spires at full resolution, inside the (upsampled) interior, specks removed. Both outputs are 0 / 1 masks.
        void full_resolution_spires(const cv::Mat &img_u8, const cv::Mat &mask_small, double threshold, cv::Mat &spires,
                                    cv::Mat &inside)
        {
            cv::Mat mask;
            cv::resize(mask_small, mask, img_u8.size(), 0, 0, cv::INTER_NEAREST);
            inside = (mask > 0) / 255;
            cv::Mat above = (img_u8 > threshold) / 255;
            cv::Mat raw;
            cv::bitwise_and(above, inside, raw);
            cv::morphologyEx(raw, spires, cv::MORPH_OPEN, cv::Mat::ones(2, 2, CV_8U));
        }

        // ------------------------------------------------------------------------------------------- 2. peaks
        // Summits of the smoothed distance to the nearest spire (low resolution), in full-resolution pixels.
        // Follows scipy.ndimage.gaussian_filter (truncate 4, reflect) and skimage.feature.peak_local_max.
        std::vector<Peak> find_peaks(const cv::Mat &spires_small, const cv::Mat &mask_small, double scale, int n_peaks)
        {
            cv::Mat not_spires, dist;
            cv::bitwise_not(spires_small, not_spires);
            cv::distanceTransform(not_spires, dist, cv::DIST_L2, 5);
            const cv::Mat valid = mask_small > 0;
            cv::Mat masked = cv::Mat::zeros(dist.size(), CV_32F);
            dist.copyTo(masked, valid);
            const double sigma = std::max(2.0, 18.0 * scale);
            const int radius = static_cast<int>(4.0 * sigma + 0.5);
            cv::Mat blurred;
            cv::GaussianBlur(masked, blurred, cv::Size(2 * radius + 1, 2 * radius + 1), sigma, sigma, cv::BORDER_REFLECT);
            cv::Mat relief = cv::Mat::zeros(dist.size(), CV_32F);
            blurred.copyTo(relief, valid);

            double lo = 0, hi = 0;
            cv::minMaxLoc(relief, &lo, &hi);
            if (hi <= 0)
                return {};

            const int size = std::max(5, static_cast<int>(35 * scale));
            const int min_distance = std::max(5, static_cast<int>(40 * scale));
            const double thr = std::max(lo, 0.15 * hi);
            cv::Mat local_max;
            cv::dilate(relief, local_max, cv::Mat::ones(size, size, CV_8U), cv::Point(size / 2, size / 2), 1,
                       cv::BORDER_REPLICATE);

            struct Candidate
            {
                int row, col;
                float height;
            };
            std::vector<Candidate> candidates;
            for (int r = min_distance; r < relief.rows - min_distance; ++r)
            {
                const float *rel = relief.ptr<float>(r);
                const float *mx = local_max.ptr<float>(r);
                for (int c = min_distance; c < relief.cols - min_distance; ++c)
                    if (rel[c] == mx[c] && rel[c] > thr)
                        candidates.push_back({r, c, rel[c]});
            }
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const Candidate &a, const Candidate &b)
                             { return a.height > b.height; });
            // highest first; a summit removes the lower ones closer than min_distance (Chebyshev)
            std::vector<Candidate> kept;
            for (const Candidate &c : candidates)
            {
                bool free_spot = true;
                for (const Candidate &k : kept)
                    if (std::max(std::abs(c.row - k.row), std::abs(c.col - k.col)) < min_distance)
                    {
                        free_spot = false;
                        break;
                    }
                if (free_spot)
                    kept.push_back(c);
            }
            if (static_cast<int>(kept.size()) > n_peaks)
                kept.resize(std::max(n_peaks, 0));
            std::vector<Peak> peaks;
            peaks.reserve(kept.size());
            for (const Candidate &c : kept)
                peaks.push_back({c.col / scale, c.row / scale, static_cast<double>(c.height), 0});
            return peaks;
        }

        // The n summits of the slice, chosen among the candidates (highest first) and given their track numbers.
        // A candidate close to a tracked summit is that summit and keeps its place; the free places go to the highest
        // other candidates, which start new tracks. Without a state, the n highest are taken and numbered by height.
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

        // ------------------------------------------------------------------------------------------- 3. crossings
        struct Crossing
        {
            double x, y, r;
        };

        struct Ray
        {
            int peak;
            int index; // index of the ray around its peak
            double angle;
            std::vector<Crossing> crossings;
        };

        // Crossings along one ray, until it leaves the interior.
        std::vector<Crossing> ray_crossings(const cv::Mat &spires, const cv::Mat &inside, double x0, double y0, double angle,
                                            double max_radius, int min_spire, int min_gap)
        {
            const int h = spires.rows, w = spires.cols;
            const double ca = std::cos(angle), sa = std::sin(angle);
            std::vector<unsigned char> s;
            std::vector<double> xs, ys;
            for (int i = 0; 1.0 + i < max_radius; ++i)
            {
                const double r = 1.0 + i;
                const double x = x0 + r * ca, y = y0 + r * sa;
                const int ix = round_half_even(x), iy = round_half_even(y);
                if (ix < 0 || ix >= w || iy < 0 || iy >= h)
                    break;
                if (!inside.at<unsigned char>(iy, ix))
                    break;
                s.push_back(spires.at<unsigned char>(iy, ix));
                xs.push_back(x);
                ys.push_back(y);
            }
            std::vector<Crossing> out;
            const int n = static_cast<int>(s.size());
            int gap = min_gap; // the start of the ray counts as void
            int k = 0;
            while (k < n)
            {
                int j = k;
                if (s[k])
                {
                    while (j < n && s[j])
                        ++j;
                    if (j - k >= min_spire && gap >= min_gap)
                    {
                        const int m = (k + j - 1) / 2;
                        out.push_back({xs[m], ys[m], 1.0 + m});
                        gap = 0;
                    }
                    // else: a sheet straight after a too-short void is the same sheet
                }
                else
                {
                    while (j < n && !s[j])
                        ++j;
                    gap = j - k;
                }
                k = j;
            }
            return out;
        }

        // The same n rays from every peak: ray i always leaves at the angle i * 2 pi / n.
        std::vector<Ray> shoot_rays(const cv::Mat &spires, const cv::Mat &inside, const std::vector<Peak> &peaks, int n,
                                    double max_radius, int min_spire, int min_gap, int threads)
        {
            std::vector<Ray> rays;
            const double step = 2.0 * PI / n;
            for (int p = 0; p < static_cast<int>(peaks.size()); ++p)
                for (int i = 0; i < n; ++i)
                    rays.push_back({p, i, i * step, {}});
            parallel_for(static_cast<int>(rays.size()), threads, [&](int i, int)
                         {
        const Peak& p = peaks[rays[i].peak];
        rays[i].crossings = ray_crossings(spires, inside, p.x, p.y, rays[i].angle, max_radius, min_spire, min_gap); });
            return rays;
        }

        // ------------------------------------------------------------------------------------------- 4. the centre
        // Sheet fragments around a centre: the spires cut along the borders of angular sectors, then split into connected
        // pieces. Only the bounding box of the spires (`roi`) is labelled; the result is indexed in roi coordinates.
        cv::Mat fragment_labels(const cv::Mat &spires, const cv::Rect &roi, double cx, double cy, int sectors, int threads)
        {
            cv::Mat cut = cv::Mat::zeros(roi.size(), CV_8U);
            auto sector_of = [&](int x, int y)
            {
                return static_cast<int>(((std::atan2(y - cy, x - cx) + PI) / (2.0 * PI)) * sectors) % sectors;
            };
            parallel_for(roi.height, threads, [&](int row, int)
                         {
        const int y = roi.y + row;
        const unsigned char* sp = spires.ptr<unsigned char>(y);
        unsigned char* out = cut.ptr<unsigned char>(row);
        int cached_x = -2, cached_sector = 0;  // sector of the last pixel computed on this row
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (!sp[x]) continue;
            const int here = sector_of(x, y);
            bool border = false;
            if (x >= 1) {
                const int left = (cached_x == x - 1) ? cached_sector : sector_of(x - 1, y);
                border = left != here;
            }
            if (!border && y >= 1) border = sector_of(x, y - 1) != here;
            cached_x = x;
            cached_sector = here;
            out[x - roi.x] = border ? 0 : 1;
        } });
            cv::Mat labels;
            cv::connectedComponents(cut, labels, 4, CV_32S);
            return labels;
        }

        // The fragment under (x, y), or the nearest one within 2 px (a crossing may sit on a sector border).
        int fragment_of(const cv::Mat &labels, const cv::Rect &roi, double x, double y)
        {
            const int ix = round_half_even(x) - roi.x, iy = round_half_even(y) - roi.y;
            int found[25];
            for (int d = 0; d < 3; ++d)
            {
                int n = 0;
                for (int yy = std::max(iy - d, 0); yy < std::min(iy + d + 1, labels.rows); ++yy)
                {
                    const int *row = labels.ptr<int>(yy);
                    for (int xx = std::max(ix - d, 0); xx < std::min(ix + d + 1, labels.cols); ++xx)
                        if (row[xx] > 0)
                            found[n++] = row[xx];
                }
                if (n == 0)
                    continue;
                std::sort(found, found + n);
                int best = found[0], best_count = 0; // most frequent label, the smallest one on a tie
                for (int i = 0; i < n;)
                {
                    int j = i;
                    while (j < n && found[j] == found[i])
                        ++j;
                    if (j - i > best_count)
                    {
                        best_count = j - i;
                        best = found[i];
                    }
                    i = j;
                }
                return best;
            }
            return 0;
        }

        // ------------------------------------------------------------------------------------------- 5. the network
        // One unknown W per fragment.
        //   count: W(f0) = target                 (ray from the centre: "this sheet is the k-th from the centre")
        //   step:  W(f1) - W(f0) = target (+-1)   (ray from another peak: two sheets crossed in a row are neighbours)
        struct Relation
        {
            bool count;
            int f0, f1;
            double target, sigma;
            int ray, k0, k1; // crossing keys: (ray, k0) and, for a step, (ray, k1)
        };

        using Solution = std::unordered_map<int, double>;

        std::vector<Relation> build_relations(const std::vector<Ray> &rays, const std::vector<Peak> &peaks, int centre,
                                              const std::vector<std::vector<int>> &frags, double sigma_count = 0.5,
                                              double sigma_step = 0.3)
        {
            const double cx = peaks[centre].x, cy = peaks[centre].y;
            std::vector<Relation> rels;
            for (int ray_id = 0; ray_id < static_cast<int>(rays.size()); ++ray_id)
            {
                const Ray &ray = rays[ray_id];
                const std::vector<int> &f = frags[ray_id];
                const int n = static_cast<int>(ray.crossings.size());
                if (ray.peak == centre)
                {
                    for (int k = 1; k <= n; ++k)
                        if (f[k - 1])
                            rels.push_back({true, f[k - 1], 0, static_cast<double>(k), sigma_count, ray_id, k - 1, -1});
                }
                else
                {
                    for (int k = 1; k < n; ++k)
                    {
                        if (!f[k - 1] || !f[k] || f[k - 1] == f[k])
                            continue;
                        const Crossing &a = ray.crossings[k - 1];
                        const Crossing &b = ray.crossings[k];
                        const bool outward = std::hypot(b.x - cx, b.y - cy) > std::hypot(a.x - cx, a.y - cy);
                        rels.push_back({false, f[k - 1], f[k], outward ? 1.0 : -1.0, sigma_step, ray_id, k - 1, k});
                    }
                }
            }
            return rels;
        }

        // Weighted least squares, solved through the normal equations (sparse Cholesky). A very weak pull towards 0 keeps
        // fragments only linked by steps (never by a count) defined, as in the Python version.
        Solution solve_lsq(const std::vector<Relation> &rels)
        {
            std::vector<int> frags;
            frags.reserve(rels.size() * 2);
            for (const Relation &r : rels)
            {
                frags.push_back(r.f0);
                if (!r.count)
                    frags.push_back(r.f1);
            }
            std::sort(frags.begin(), frags.end());
            frags.erase(std::unique(frags.begin(), frags.end()), frags.end());
            const int n = static_cast<int>(frags.size());
            std::unordered_map<int, int> col;
            col.reserve(frags.size() * 2);
            for (int i = 0; i < n; ++i)
                col[frags[i]] = i;

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(rels.size() * 4 + n);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
            for (const Relation &r : rels)
            {
                const double w2 = 1.0 / (r.sigma * r.sigma);
                if (r.count)
                {
                    const int a = col[r.f0];
                    triplets.emplace_back(a, a, w2);
                    rhs[a] += w2 * r.target;
                }
                else
                {
                    const int a = col[r.f0], b = col[r.f1];
                    triplets.emplace_back(a, a, w2);
                    triplets.emplace_back(b, b, w2);
                    triplets.emplace_back(a, b, -w2);
                    triplets.emplace_back(b, a, -w2);
                    rhs[b] += w2 * r.target;
                    rhs[a] -= w2 * r.target;
                }
            }
            const double regularisation = 1e-3;
            for (int i = 0; i < n; ++i)
                triplets.emplace_back(i, i, regularisation * regularisation);
            Eigen::SparseMatrix<double> normal(n, n);
            normal.setFromTriplets(triplets.begin(), triplets.end());

            Eigen::VectorXd w;
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(normal);
            if (ldlt.info() == Eigen::Success)
                w = ldlt.solve(rhs);
            if (ldlt.info() != Eigen::Success)
            {
                Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg(normal);
                cg.setTolerance(1e-12);
                w = cg.solve(rhs);
            }
            Solution solution;
            solution.reserve(frags.size() * 2);
            for (int i = 0; i < n; ++i)
                solution[frags[i]] = w[i];
            return solution;
        }

        double residual(const Relation &r, const Solution &W)
        {
            return r.count ? W.at(r.f0) - r.target : W.at(r.f1) - W.at(r.f0) - r.target;
        }

        // One peak tried as the centre: its fragments under every crossing, its relations, its solution and its tension
        // (mean squared pull of the whole network; lower: more consistent).
        struct Trial
        {
            std::vector<std::vector<int>> frags;
            std::vector<Relation> rels;
            Solution W;
            double tension = std::numeric_limits<double>::infinity();
        };

        Trial try_centre(const std::vector<Ray> &rays, const std::vector<Peak> &peaks, int centre, const cv::Mat &spires,
                         const cv::Rect &roi, int sectors, int threads)
        {
            Trial trial;
            {
                const cv::Mat labels = fragment_labels(spires, roi, peaks[centre].x, peaks[centre].y, sectors, threads);
                trial.frags.resize(rays.size());
                for (size_t i = 0; i < rays.size(); ++i)
                {
                    trial.frags[i].reserve(rays[i].crossings.size());
                    for (const Crossing &c : rays[i].crossings)
                        trial.frags[i].push_back(fragment_of(labels, roi, c.x, c.y));
                }
            } // the label map is released here: only the fragment ids under the crossings are kept
            trial.rels = build_relations(rays, peaks, centre, trial.frags);
            if (trial.rels.empty())
                return trial;
            trial.W = solve_lsq(trial.rels);
            bool has_count = false;
            for (const Relation &r : trial.rels)
                has_count = has_count || r.count;
            if (has_count)
            {
                double sum = 0;
                for (const Relation &r : trial.rels)
                {
                    const double pull = residual(r, trial.W) / r.sigma;
                    sum += pull * pull;
                }
                trial.tension = sum / trial.rels.size();
            }
            return trial;
        }

        // The centre is the peak whose numbering leaves the whole network least tense. Scores are 1 / (1 + mean squared
        // pull). Along z, the previous centre is kept while its score stays within `keep` of the best.
        int choose_centre(const std::vector<Peak> &peaks, const std::vector<double> &scores, const cv::Point2d *previous,
                          double keep = 0.9, double radius = 60.0)
        {
            int best = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
            if (previous)
            {
                int near = -1;
                for (int p = 0; p < static_cast<int>(peaks.size()); ++p)
                    if (std::hypot(peaks[p].x - previous->x, peaks[p].y - previous->y) < radius &&
                        (near < 0 || scores[p] > scores[near]))
                        near = p;
                if (near >= 0 && scores[near] >= keep * scores[best])
                    best = near;
            }
            return best;
        }

        // ------------------------------------------------------------------------------------------- JSON
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

    // ------------------------------------------------------------------------------------------- data
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

    // ------------------------------------------------------------------------------------------- one slice
    SliceResult process_slice(const cv::Mat &img_u8, const Params &params, const cv::Point2d *previous_centre,
                              TrackState *tracks)
    {
        CV_Assert(img_u8.type() == CV_8U);
        SliceResult result;
        const int threads = params.threads > 0 ? params.threads : hardware_threads();
        const int h = img_u8.rows, w = img_u8.cols;

        auto t0 = Clock::now();
        cv::Mat small, mask_small, spires_small;
        cv::resize(img_u8, small, cv::Size(), params.scale, params.scale, cv::INTER_AREA);
        double threshold = 0;
        detect_scroll_interior_mask(small, params.scale, params.frame_px, params.close_px, mask_small, spires_small,
                                    threshold);
        result.timings.mask = seconds_since(t0);

        t0 = Clock::now();
        cv::Mat spires, inside;
        full_resolution_spires(img_u8, mask_small, threshold, spires, inside);
        result.timings.spires = seconds_since(t0);

        t0 = Clock::now();
        // twice as many candidates as summits kept, so that a tracked summit that lost a few ranks is still found
        const std::vector<Peak> candidates = find_peaks(spires_small, mask_small, params.scale, 2 * params.peaks);
        const std::vector<Peak> peaks =
            select_peaks(candidates, params.peaks, tracks, params.track_radius, params.track_memory);
        result.timings.peaks = seconds_since(t0);
        result.n_rays = std::max(params.rays, 1);
        if (peaks.empty())
            return result;

        t0 = Clock::now();
        const double max_radius = params.max_radius > 0 ? params.max_radius : std::max(h, w);
        const std::vector<Ray> rays =
            shoot_rays(spires, inside, peaks, result.n_rays, max_radius, params.min_spire, params.min_gap, threads);
        result.timings.rays = seconds_since(t0);

        // every peak is tried as the centre; the trials run side by side
        t0 = Clock::now();
        cv::Rect roi = cv::boundingRect(spires);
        if (roi.area() == 0)
            roi = cv::Rect(0, 0, 1, 1);
        const int n_peaks = static_cast<int>(peaks.size());
        const int jobs = std::max(1, std::min({params.centre_jobs, threads, n_peaks}));
        const int inner_threads = std::max(1, threads / jobs);
        std::vector<Trial> trials(n_peaks);
        parallel_for(n_peaks, jobs, [&](int p, int)
                     { trials[p] = try_centre(rays, peaks, p, spires, roi, params.sectors, inner_threads); });
        std::vector<double> scores(n_peaks);
        for (int p = 0; p < n_peaks; ++p)
            scores[p] = 1.0 / (1.0 + trials[p].tension);
        const int centre = choose_centre(peaks, scores, previous_centre);
        result.timings.centre = seconds_since(t0);

        t0 = Clock::now();
        const Trial &trial = trials[centre];
        if (trial.rels.empty())
            return result;

        // pull of every crossing: the largest |residual / sigma| among the relations it takes part in
        std::vector<std::vector<double>> pull(rays.size());
        for (size_t i = 0; i < rays.size(); ++i)
            pull[i].assign(rays[i].crossings.size(), 0.0);
        for (const Relation &r : trial.rels)
        {
            const double p = std::abs(residual(r, trial.W) / r.sigma);
            pull[r.ray][r.k0] = std::max(pull[r.ray][r.k0], p);
            if (r.k1 >= 0)
                pull[r.ray][r.k1] = std::max(pull[r.ray][r.k1], p);
        }

        for (int ray_id = 0; ray_id < static_cast<int>(rays.size()); ++ray_id)
        {
            const Ray &ray = rays[ray_id];
            LineOut line;
            line.ray_index = ray.index;
            line.peak_index = ray.peak;
            line.track = peaks[ray.peak].track;
            line.angle = ray.angle;
            for (size_t k = 0; k < ray.crossings.size(); ++k)
            {
                const int f = trial.frags[ray_id][k];
                const auto it = trial.W.find(f);
                if (it == trial.W.end())
                    continue;
                const double u = it->second;
                line.crossings.push_back(
                    {ray.crossings[k].x, ray.crossings[k].y, round_half_even(u), u, f, pull[ray_id][k]});
            }
            if (!line.crossings.empty())
                result.lines.push_back(std::move(line));
        }
        result.has_centre = true;
        result.cx = peaks[centre].x;
        result.cy = peaks[centre].y;
        result.centre_peak = centre;
        result.centre_track = peaks[centre].track;
        result.peaks = peaks;
        result.scores = scores;
        result.timings.output = seconds_since(t0);
        return result;
    }

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
            s += ",\"track\":" + std::to_string(result.centre_track) + "}";
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
