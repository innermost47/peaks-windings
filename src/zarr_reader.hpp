// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pw
{

   // The position of a chunk in the chunk grid of a volume (z, y, x).
   struct ChunkIndex
   {
      int64_t z = 0, y = 0, x = 0;
   };

   class ZarrVolume
   {
   public:
      static std::unique_ptr<ZarrVolume> open(const std::string &url, int connections = 32, int threads = 0,
                                              double cache_gb = 2.0);
      virtual ~ZarrVolume() = default;

      virtual const std::vector<int64_t> &shape() const = 0;
      virtual const std::vector<int64_t> &chunks() const = 0;
      virtual std::string description() const = 0;
      virtual int cv_type() const = 0; // the OpenCV type of one voxel (CV_8UC1, CV_16UC1...)
      virtual cv::Mat read_slice(int64_t z) = 0;
      virtual std::vector<cv::Mat> read_region(int64_t z0, int64_t z1, int64_t y0, int64_t y1, int64_t x0,
                                               int64_t x1) = 0;
      // The given chunks, fetched together and decoded: one 3-D cv::Mat (z, y, x) of the full chunk shape each, all
      // fill value where a chunk does not exist. Nothing is cached: the caller keeps what it needs.
      virtual std::vector<cv::Mat> read_chunks(const std::vector<ChunkIndex> &chunks) = 0;
   };

} // namespace pw
