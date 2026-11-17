// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once

#include <climits>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pw
{

    struct Point
    {
        double x = 0, y = 0;
        int z = 0, ray = 0, winding = 0;
        int track = 0;    // the summit the ray leaves from (0 for result files without tracks)
        float pull = 0.f; // how far the network had to pull the point to agree with the others, in sigmas
    };

    // The centre a slice's labels are counted from (the "centre" of its result file).
    struct Centre
    {
        double x = 0, y = 0;
    };

    // The result files (scroll*_z*.json) of a folder, sorted by name. Throws when there is none.
    std::vector<std::string> result_files(const std::string &results_dir);

    // Number of rays of the scan (stored in each result file; 360 for files without the field).
    int ray_count(const std::string &results_dir);

    // All points of all slices, in file order. Only the slices with z_min <= z <= z_max are kept.
    std::vector<Point> load_points(const std::string &results_dir, int z_min = INT_MIN, int z_max = INT_MAX,
                                   int threads = 0);

    // Relabels the points in place so that one sheet keeps one label along Z. Returns the correction applied to each
    // slice ({z: label shift subtracted}).
    std::map<int, int> align_labels(std::vector<Point> &points);

    // ------------------------------------------------------------------------------------------- streamed loading
    // load_points keeps every point of every layer, which does not fit in memory for a whole scroll. The two functions
    // below give the same result in two passes over the files, holding only two consecutive slices at a time:
    //   1. survey_layers aligns the labels slice by slice (exactly as align_labels) and only counts the points;
    //   2. load_layer reads the files again and keeps only the points of the chosen layer (and summits).

    // What one pass over the result files tells about the layers, without keeping their points.
    struct LayerSurvey
    {
        std::map<int, int> correction;                     // {z: label shift subtracted}, as align_labels returns it
        std::vector<int> layer_order;                      // aligned labels, in order of first appearance in the files
        std::map<int, size_t> layer_points;                // aligned label -> number of points
        std::map<int, std::map<int, size_t>> track_points; // aligned label -> {track: number of points}
        std::map<int, Centre> centres;                     // z -> the centre of the slice (when its file gives one)
    };

    // Pass 1. Prints the same alignment line as align_labels. Files whose name does not tell their z cannot be taken in
    // Z order before being read: the survey is then made from load_points (everything in memory, as before).
    LayerSurvey survey_layers(const std::string &results_dir, int z_min = INT_MIN, int z_max = INT_MAX, int threads = 0);

    // Pass 2. The points of one aligned layer (labels already corrected), in file order, as load_points followed by
    // align_labels would give them. With `tracks` not empty, only the points of those summits are kept.
    std::vector<Point> load_layer(const std::string &results_dir, int z_min, int z_max, int threads,
                                  const LayerSurvey &survey, int layer, const std::vector<int> &tracks = {});

    // Pass 2, slice by slice: calls fn(z, points) for every slice holding points of the layer (labels corrected), in Z
    // order, without keeping the layer in memory. With `tracks` not empty, only the points of those summits.
    void for_each_layer_slice(const std::string &results_dir, int z_min, int z_max, int threads,
                              const LayerSurvey &survey, int layer, const std::vector<int> &tracks,
                              const std::function<void(int, const std::vector<Point> &)> &fn);

} // namespace pw
