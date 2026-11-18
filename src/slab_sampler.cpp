// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "slab_sampler.hpp"

#include "chunk_mosaic.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace pw
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

        bool on_sheet(const Rail &r, int k)
        {
            return std::isfinite(r.xs[k]) && std::isfinite(r.ys[k]) && std::isfinite(r.nx[k]) && std::isfinite(r.ny[k]);
        }

        // The point `off` voxels across the sheet from column k, kept where a bilinear sample stays inside the volume.
        cv::Point2d across(const Rail &r, int k, int off, const ChunkMosaic &m)
        {
            const double x = std::min(std::max(r.xs[k] + off * r.nx[k], 0.0), m.width() - 1.001);
            const double y = std::min(std::max(r.ys[k] + off * r.ny[k], 0.0), m.height() - 1.001);
            return {x, y};
        }

        // Adds to `tiles` the tiles of every voxel the slab of column k reads (the segment across the sheet and the
        // bilinear neighbours of its points: the box of its two clamped ends, one voxel wider).
        void add_slab_tiles(const Rail &r, int k, int half_thick, const ChunkMosaic &m, std::vector<int> &tiles)
        {
            const cv::Point2d a = across(r, k, -half_thick, m), b = across(r, k, half_thick, m);
            const int64_t x0 = static_cast<int64_t>(std::min(a.x, b.x)), y0 = static_cast<int64_t>(std::min(a.y, b.y));
            const int64_t x1 = std::min<int64_t>(static_cast<int64_t>(std::max(a.x, b.x)) + 1, m.width() - 1);
            const int64_t y1 = std::min<int64_t>(static_cast<int64_t>(std::max(a.y, b.y)) + 1, m.height() - 1);
            const int t00 = m.tile_of(y0, x0), t01 = m.tile_of(y0, x1), t10 = m.tile_of(y1, x0), t11 = m.tile_of(y1, x1);
            tiles.push_back(t00);
            if (t01 != t00)
                tiles.push_back(t01);
            if (t10 != t00)
                tiles.push_back(t10);
            if (t11 != t01 && t11 != t10)
                tiles.push_back(t11);
        }

        template <class T>
        void sample_columns(const ChunkMosaic &m, const Rail &r, int half_thick, int z, int k0, int k1, double *row,
                            cv::Vec3f *where)
        {
            for (int k = k0; k < k1; ++k)
            {
                if (!on_sheet(r, k))
                    continue;
                double best = 0;
                for (int off = -half_thick; off <= half_thick; ++off)
                {
                    const cv::Point2d p = across(r, k, off, m);
                    const int64_t x0 = static_cast<int64_t>(p.x), y0 = static_cast<int64_t>(p.y);
                    const double ax = p.x - x0, ay = p.y - y0;
                    const int64_t x1 = std::min<int64_t>(x0 + 1, m.width() - 1), y1 = std::min<int64_t>(y0 + 1, m.height() - 1);
                    const double v = m.voxel<T>(z, y0, x0) * (1 - ax) * (1 - ay) + m.voxel<T>(z, y0, x1) * ax * (1 - ay) +
                                     m.voxel<T>(z, y1, x0) * (1 - ax) * ay + m.voxel<T>(z, y1, x1) * ax * ay;
                    best = off == -half_thick ? v : std::max(best, v);
                }
                row[k] = std::isfinite(best) ? best : 0.0;
                where[k] = cv::Vec3f(static_cast<float>(r.xs[k]), static_cast<float>(r.ys[k]), static_cast<float>(z));
            }
        }

        void sample_columns(const ChunkMosaic &m, const Rail &r, int half_thick, int z, int k0, int k1, double *row,
                            cv::Vec3f *where)
        {
            switch (CV_MAT_DEPTH(m.cv_type()))
            {
            case CV_8U: sample_columns<unsigned char>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_8S: sample_columns<signed char>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_16U: sample_columns<unsigned short>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_16S: sample_columns<short>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_32S: sample_columns<int>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_32F: sample_columns<float>(m, r, half_thick, z, k0, k1, row, where); break;
            case CV_64F: sample_columns<double>(m, r, half_thick, z, k0, k1, row, where); break;
            default: throw std::runtime_error("unsupported voxel type");
            }
        }
    } // namespace

    SlabImage sample_slab(const std::vector<int> &dense_z, int width, const std::function<Rail(int)> &rail,
                          const SlabParams &o, ZarrVolume &volume)
    {
        if (dense_z.empty())
            return {};
        if (CV_MAT_CN(volume.cv_type()) != 1)
            throw std::runtime_error("the volume must have a single channel");
        const int threads = o.threads > 0 ? o.threads : hardware_threads();
        const auto t0 = Clock::now();
        const int half_thick = o.slab / 2;

        ChunkMosaic mosaic(volume);
        const int64_t depth = mosaic.chunk_depth();
        const size_t budget = std::max<size_t>(1, static_cast<size_t>(o.cache_mb * 1048576.0 / mosaic.chunk_bytes()));
        std::printf("    - Reading only the chunks the slab crosses, up to %zu decoded chunk(s) (%.0f MB) at once\n",
                    budget, budget * mosaic.chunk_bytes() / 1048576.0);
        std::fflush(stdout);

        cv::Mat image = cv::Mat::zeros(static_cast<int>(dense_z.size()), width, CV_64F);
        // where every pixel was taken in the volume: (x, y, z), NaN where nothing was sampled
        cv::Mat coords(image.size(), CV_32FC3, cv::Scalar::all(std::numeric_limits<float>::quiet_NaN()));

        std::vector<int> stamp(static_cast<size_t>(mosaic.tile_count()), -1);
        int run_id = 0;
        for (size_t i0 = 0; i0 < dense_z.size();)
        {
            // the slices of one layer of the chunk grid, and their rails
            const int64_t layer = dense_z[i0] / depth;
            size_t i1 = i0;
            while (i1 < dense_z.size() && dense_z[i1] / depth == layer)
                ++i1;
            const int n = static_cast<int>(i1 - i0);
            std::vector<Rail> rails(n);
            parallel_for(n, threads, [&](int i, int)
                         { rails[i] = rail(dense_z[i0 + i]); });

            // the tiles every column reads over these slices
            std::vector<std::vector<int>> needs(width);
            parallel_for(width, threads, [&](int k, int)
                         {
                std::vector<int>& t = needs[k];
                for (const Rail& r : rails)
                    if (on_sheet(r, k)) add_slab_tiles(r, k, half_thick, mosaic, t);
                std::sort(t.begin(), t.end());
                t.erase(std::unique(t.begin(), t.end()), t.end()); });

            // runs of consecutive columns whose tiles fit the budget, each read then sampled
            std::vector<int> run;
            int k0 = 0;
            auto sample_run = [&](int k1)
            {
                if (!run.empty())
                {
                    mosaic.hold(layer, run);
                    parallel_for(n, threads, [&](int i, int)
                                 {
                        const int r = static_cast<int>(i0) + i;
                        sample_columns(mosaic, rails[i], half_thick, dense_z[i0 + i], k0, k1, image.ptr<double>(r),
                                       coords.ptr<cv::Vec3f>(r)); });
                }
                run.clear();
                ++run_id;
                k0 = k1;
            };
            for (int k = 0; k < width; ++k)
            {
                size_t fresh = 0;
                for (int t : needs[k])
                    fresh += stamp[t] != run_id;
                if (!run.empty() && run.size() + fresh > budget)
                    sample_run(k);
                for (int t : needs[k])
                    if (stamp[t] != run_id)
                        stamp[t] = run_id, run.push_back(t);
            }
            sample_run(width);

            std::printf("    Sampled slices: %zu/%zu (%.0f s, %zu chunk(s) read, %.1f GB)\n", i1, dense_z.size(),
                        seconds_since(t0), mosaic.chunks_read(),
                        mosaic.chunks_read() * mosaic.chunk_bytes() / 1073741824.0);
            std::fflush(stdout);
            i0 = i1;
        }
        return {image, coords};
    }
} // namespace pw
