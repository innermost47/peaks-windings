// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include "zarr_reader.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pw
{
    // The decoded chunks of one layer of the chunk grid (one chunk deep along Z) that a reader needs, addressed by
    // tile (the position of a chunk in that layer). Asking for a new set of tiles reads only the chunks not held yet
    // and frees those no longer needed, so two consecutive sets that overlap share their common chunks.
    class ChunkMosaic
    {
    public:
        explicit ChunkMosaic(ZarrVolume &volume);

        int64_t chunk_depth() const { return depth_; }
        int64_t height() const { return height_; } // of the volume, in voxels along y
        int64_t width() const { return width_; }   // of the volume, in voxels along x
        int cv_type() const { return type_; }
        size_t chunk_bytes() const { return chunk_bytes_; }
        int tile_count() const { return static_cast<int>(held_.size()); }
        size_t chunks_read() const { return chunks_read_; } // since the mosaic was made

        // The tile of the chunk holding the voxel column (y, x), which must lie inside the volume.
        int tile_of(int64_t y, int64_t x) const { return static_cast<int>((y / rows_) * tiles_x_ + x / cols_); }

        // Holds exactly the chunks of `tiles` in the chunk layer `layer` (z from layer * chunk_depth()): the missing
        // ones are read together, the others are freed. Returns how many chunks were read.
        size_t hold(int64_t layer, const std::vector<int> &tiles);

        // The voxel (z, y, x) as a number; T is the C++ type of cv_type(). z must lie in the layer held, and the
        // tile of (y, x) must have been asked for (0 otherwise).
        template <class T>
        double voxel(int64_t z, int64_t y, int64_t x) const
        {
            const cv::Mat &chunk = held_[tile_of(y, x)];
            if (chunk.empty())
                return 0;
            return static_cast<double>(
                chunk.ptr<T>(static_cast<int>(z - z0_), static_cast<int>(y % rows_))[x % cols_]);
        }

    private:
        ZarrVolume &volume_;
        int64_t depth_ = 1, rows_ = 1, cols_ = 1; // chunk shape (z, y, x)
        int64_t height_ = 0, width_ = 0;
        int64_t tiles_x_ = 0;
        int type_ = 0;
        size_t chunk_bytes_ = 0;
        int64_t layer_ = -1, z0_ = 0;
        std::vector<cv::Mat> held_; // one per tile, empty when not held
        size_t chunks_read_ = 0;
    };
} // namespace pw
