// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "chunk_mosaic.hpp"

#include <stdexcept>
#include <string>

namespace pw
{
    ChunkMosaic::ChunkMosaic(ZarrVolume &volume) : volume_(volume)
    {
        const auto &shape = volume.shape();
        const auto &chunks = volume.chunks();
        depth_ = static_cast<int64_t>(chunks[0]);
        rows_ = static_cast<int64_t>(chunks[1]);
        cols_ = static_cast<int64_t>(chunks[2]);
        if (depth_ <= 0 || rows_ <= 0 || cols_ <= 0)
            throw std::runtime_error("invalid chunk shape");
        height_ = static_cast<int64_t>(shape[1]);
        width_ = static_cast<int64_t>(shape[2]);
        tiles_x_ = (width_ + cols_ - 1) / cols_;
        const int64_t tiles_y = (height_ + rows_ - 1) / rows_;
        type_ = volume.cv_type();
        chunk_bytes_ = static_cast<size_t>(depth_ * rows_ * cols_) * CV_ELEM_SIZE(type_);
        held_.resize(static_cast<size_t>(tiles_y * tiles_x_));
    }

    size_t ChunkMosaic::hold(int64_t layer, const std::vector<int> &tiles)
    {
        if (layer != layer_)
        {
            for (cv::Mat &chunk : held_)
                chunk.release();
            layer_ = layer;
            z0_ = layer * depth_;
        }
        std::vector<char> wanted(held_.size(), 0);
        for (int t : tiles)
        {
            if (t < 0 || t >= tile_count())
                throw std::out_of_range("tile " + std::to_string(t) + " is outside the chunk grid");
            wanted[t] = 1;
        }
        // free first, so what is held never exceeds the new set
        std::vector<ChunkIndex> missing;
        std::vector<int> missing_tiles;
        for (int t = 0; t < tile_count(); ++t)
        {
            if (!wanted[t])
                held_[t].release();
            else if (held_[t].empty())
            {
                missing.push_back({layer, t / tiles_x_, t % tiles_x_});
                missing_tiles.push_back(t);
            }
        }
        if (missing.empty())
            return 0;
        std::vector<cv::Mat> read = volume_.read_chunks(missing);
        for (size_t i = 0; i < read.size(); ++i)
            held_[missing_tiles[i]] = std::move(read[i]);
        chunks_read_ += missing.size();
        return missing.size();
    }
} // namespace pw
