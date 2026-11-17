// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "layers.hpp"

#include "parallel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <set>
#include <unordered_map>

namespace pw {
namespace {

using json = nlohmann::json;

const int MIN_VOTES = 20;      // fewer matched points than this between two slices: no shift is applied
const double MIN_SHARE = 0.5;  // the most common difference must hold for at least half of the matches

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The z written in a file name (..._z01500.json), or false when the name does not tell.
bool z_of_name(const std::string& path, long long& z) {
    const std::string name = std::filesystem::path(path).filename().string();
    const size_t dot = name.rfind(".json");
    const size_t tag = name.rfind("_z");
    if (dot == std::string::npos || tag == std::string::npos || tag + 2 >= dot) return false;
    const std::string digits = name.substr(tag + 2, dot - tag - 2);
    if (digits.find_first_not_of("0123456789") != std::string::npos) return false;
    z = std::atoll(digits.c_str());
    return true;
}

// A ray is one index around one summit.
long long ray_key(const Point& p) { return (static_cast<long long>(p.track) << 32) | static_cast<unsigned>(p.ray); }

// Points of one slice, grouped by ray in order of first appearance (as a Python dict would).
struct ByRay {
    std::vector<long long> rays;
    std::unordered_map<long long, std::vector<size_t>> points;

    explicit ByRay(const std::vector<Point>& slice) {
        for (size_t i = 0; i < slice.size(); ++i) {
            const long long key = ray_key(slice[i]);
            auto it = points.find(key);
            if (it == points.end()) {
                rays.push_back(key);
                it = points.emplace(key, std::vector<size_t>()).first;
            }
            it->second.push_back(i);
        }
    }
};

struct Shift {
    int shift = 0;
    int votes = 0;
    double share = 0;
};

// Label shift between two slices: b's labels minus a's labels for the same sheet.
Shift slice_shift(const std::vector<Point>& a, const std::vector<Point>& b) {
    const ByRay by_a(a), by_b(b);
    std::vector<std::pair<int, int>> votes;  // (difference, count), in order of first appearance
    for (long long ray : by_a.rays) {
        const std::vector<size_t>& pa = by_a.points.at(ray);
        const auto found = by_b.points.find(ray);
        if (found == by_b.points.end() || pa.size() < 2) continue;
        const std::vector<size_t>& pb = found->second;
        // spacing between sheets on this ray: the matched point must be much closer than that
        std::vector<size_t> order(pa);
        std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return a[i].winding < a[j].winding; });
        std::vector<double> gaps;
        for (size_t k = 0; k + 1 < order.size(); ++k)
            gaps.push_back(std::hypot(a[order[k + 1]].x - a[order[k]].x, a[order[k + 1]].y - a[order[k]].y));
        std::sort(gaps.begin(), gaps.end());
        const size_t n = gaps.size();
        const double median = n % 2 ? gaps[n / 2] : 0.5 * (gaps[n / 2 - 1] + gaps[n / 2]);
        const double tol = 0.4 * median;
        for (size_t i : pa) {
            double best = 1e300;
            size_t best_j = pb[0];
            for (size_t j : pb) {
                const double d = std::hypot(b[j].x - a[i].x, b[j].y - a[i].y);
                if (d < best) {
                    best = d;
                    best_j = j;
                }
            }
            if (best < tol) {
                const int diff = b[best_j].winding - a[i].winding;
                auto it = std::find_if(votes.begin(), votes.end(), [&](const auto& v) { return v.first == diff; });
                if (it == votes.end())
                    votes.emplace_back(diff, 1);
                else
                    ++it->second;
            }
        }
    }
    Shift result;
    int best_count = 0;
    for (const auto& [diff, count] : votes) {
        result.votes += count;
        if (count > best_count) {
            best_count = count;
            result.shift = diff;
        }
    }
    if (result.votes == 0) return Shift();
    result.share = static_cast<double>(best_count) / result.votes;
    return result;
}

// The shift actually applied between two consecutive slices (0 when the match is not trusted).
int trusted_shift(const std::vector<Point>& a, const std::vector<Point>& b, int& unsure) {
    const Shift s = slice_shift(a, b);
    if (s.votes < MIN_VOTES || s.share < MIN_SHARE) {
        ++unsure;
        return 0;
    }
    return s.shift;
}

void print_alignment(int moved, int unsure) {
    std::printf("--> Labels aligned along Z: %d slice transition(s) corrected, %d without enough matches.\n", moved,
                unsure);
}

// The points of one result file with z_min <= z <= z_max, in file order (and its centre, when it gives one).
std::vector<Point> read_file(const std::string& path, int z_min, int z_max, Centre* centre = nullptr,
                             bool* has_centre = nullptr) {
    std::vector<Point> out;
    const json data = json::parse(read_text(path));
    if (has_centre) {
        *has_centre = false;
        const auto c = data.find("centre");
        if (c != data.end() && c->is_object() && c->contains("x") && c->contains("y") && (*c)["x"].is_number() &&
            (*c)["y"].is_number()) {
            centre->x = (*c)["x"].get<double>();
            centre->y = (*c)["y"].get<double>();
            *has_centre = true;
        }
    }
    const bool has_z = data.contains("z") && data["z"].is_number();
    const int file_z = has_z ? data["z"].get<int>() : 0;
    if (!data.contains("lines")) return out;
    for (const json& line : data["lines"]) {
        const int ray = line.value("ray_index", 0);
        const int track = line.value("track", 0);
        if (!line.contains("crossings")) continue;
        for (const json& pt : line["crossings"]) {
            Point p;
            p.x = pt.at("x").get<double>();
            p.y = pt.at("y").get<double>();
            p.z = has_z ? file_z : pt.at("z").get<int>();
            p.ray = ray;
            p.track = track;
            p.winding = pt.at("winding_label").get<int>();
            const auto pull = pt.find("pull");
            p.pull = pull != pt.end() && pull->is_number() ? pull->get<float>() : 0.f;
            if (p.z >= z_min && p.z <= z_max) out.push_back(p);
        }
    }
    return out;
}

// The result files kept by the z range of their name (files whose name does not tell are kept), in name order.
std::vector<std::string> files_in_range(const std::string& results_dir, int z_min, int z_max) {
    std::vector<std::string> files;
    for (const std::string& path : result_files(results_dir)) {
        long long z = 0;
        if (z_of_name(path, z) && (z < z_min || z > z_max)) continue;  // skipped without being read
        files.push_back(path);
    }
    return files;
}

// Adds the points of one aligned slice to the counts. `rank` orders the files by name, `index` the points in it.
struct FirstSeen {
    size_t rank = 0, index = 0;
    bool operator<(const FirstSeen& o) const { return rank != o.rank ? rank < o.rank : index < o.index; }
};

void count_points(LayerSurvey& survey, std::map<int, FirstSeen>& first, const std::vector<Point>& slice, int shift,
                  size_t rank, size_t offset) {
    for (size_t i = 0; i < slice.size(); ++i) {
        const int label = slice[i].winding - shift;
        const FirstSeen here{rank, offset + i};
        const auto it = first.find(label);
        if (it == first.end())
            first.emplace(label, here);
        else if (here < it->second)
            it->second = here;
        ++survey.layer_points[label];
        ++survey.track_points[label][slice[i].track];
    }
}

void order_layers(LayerSurvey& survey, const std::map<int, FirstSeen>& first) {
    std::vector<std::pair<FirstSeen, int>> order;
    for (const auto& [label, seen] : first) order.emplace_back(seen, label);
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    survey.layer_order.clear();
    for (const auto& item : order) survey.layer_order.push_back(item.second);
}

// The survey of points already in memory and already aligned (the fallback for files without z in their name).
LayerSurvey survey_of(const std::vector<Point>& points, std::map<int, int> correction) {
    LayerSurvey survey;
    survey.correction = std::move(correction);
    std::map<int, FirstSeen> first;
    count_points(survey, first, points, 0, 0, 0);
    order_layers(survey, first);
    return survey;
}

}  // namespace

std::vector<std::string> result_files(const std::string& results_dir) {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(results_dir, ec)) {
        const std::string name = entry.path().filename().string();
        // the pattern scroll*_z*.json
        if (name.size() < 14 || name.compare(0, 6, "scroll") != 0) continue;
        if (name.compare(name.size() - 5, 5, ".json") != 0) continue;
        const size_t tag = name.find("_z", 6);
        if (tag == std::string::npos || tag + 2 > name.size() - 5) continue;
        files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("[ERROR] No result files (scroll*_z*.json) found in '" + results_dir + "'.");
    return files;
}

int ray_count(const std::string& results_dir) {
    const json data = json::parse(read_text(result_files(results_dir)[0]));
    return data.value("n_rays", 360);
}

std::vector<Point> load_points(const std::string& results_dir, int z_min, int z_max, int threads) {
    const std::vector<std::string> files = files_in_range(results_dir, z_min, z_max);
    std::vector<std::vector<Point>> per_file(files.size());
    parallel_for(static_cast<int>(files.size()), threads > 0 ? threads : hardware_threads(),
                 [&](int i, int) { per_file[i] = read_file(files[i], z_min, z_max); });
    size_t total = 0;
    for (const auto& v : per_file) total += v.size();
    std::vector<Point> points;
    points.reserve(total);
    for (auto& v : per_file) {
        points.insert(points.end(), v.begin(), v.end());
        std::vector<Point>().swap(v);
    }
    return points;
}

std::map<int, int> align_labels(std::vector<Point>& points) {
    std::map<int, std::vector<size_t>> by_z;
    for (size_t i = 0; i < points.size(); ++i) by_z[points[i].z].push_back(i);
    std::map<int, int> correction;
    int unsure = 0, moved = 0;
    if (!by_z.empty()) correction[by_z.begin()->first] = 0;
    auto slice_of = [&](const std::vector<size_t>& indices) {
        std::vector<Point> slice;
        slice.reserve(indices.size());
        for (size_t i : indices) slice.push_back(points[i]);
        return slice;
    };
    std::vector<Point> current = by_z.empty() ? std::vector<Point>() : slice_of(by_z.begin()->second);
    for (auto a = by_z.begin(); a != by_z.end(); ++a) {
        const auto b = std::next(a);
        if (b == by_z.end()) break;
        std::vector<Point> next = slice_of(b->second);
        const int shift = trusted_shift(current, next, unsure);
        correction[b->first] = correction[a->first] + shift;
        moved += shift != 0;
        current = std::move(next);
    }
    for (Point& p : points) p.winding -= correction[p.z];
    print_alignment(moved, unsure);
    return correction;
}

LayerSurvey survey_layers(const std::string& results_dir, int z_min, int z_max, int threads) {
    const std::vector<std::string> files = files_in_range(results_dir, z_min, z_max);
    // the files in Z order (name order among the files of one z), each with its rank in name order
    std::vector<std::pair<long long, size_t>> by_z;
    for (size_t r = 0; r < files.size(); ++r) {
        long long z = 0;
        if (!z_of_name(files[r], z)) {  // cannot be ordered before being read: everything in memory, as before
            std::vector<Point> points = load_points(results_dir, z_min, z_max, threads);
            std::map<int, int> correction = align_labels(points);
            return survey_of(points, std::move(correction));
        }
        by_z.emplace_back(z, r);
    }
    std::stable_sort(by_z.begin(), by_z.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    const int workers = threads > 0 ? threads : hardware_threads();
    const size_t batch = static_cast<size_t>(std::max(1, 2 * workers));  // files read at the same time
    LayerSurvey survey;
    std::map<int, FirstSeen> first;
    int unsure = 0, moved = 0;
    bool has_previous = false;
    int previous_z = 0;
    std::vector<Point> previous;  // the last slice seen (raw labels), the only one kept between batches

    // A slice is all the points of one z; the files of one z follow each other in `by_z`.
    std::vector<Point> current;
    int current_z = 0;
    bool has_current = false;
    std::vector<std::pair<size_t, size_t>> current_parts;  // (rank, number of points) of the files of the slice
    auto close_slice = [&]() {
        if (!has_current) return;
        int shift = 0;
        if (!has_previous) {
            survey.correction[current_z] = 0;
        } else {
            const int step = trusted_shift(previous, current, unsure);
            moved += step != 0;
            shift = survey.correction[previous_z] + step;
            survey.correction[current_z] = shift;
        }
        size_t offset = 0;
        for (const auto& [rank, count] : current_parts) {
            const std::vector<Point> part(current.begin() + static_cast<std::ptrdiff_t>(offset),
                                          current.begin() + static_cast<std::ptrdiff_t>(offset + count));
            count_points(survey, first, part, shift, rank, 0);
            offset += count;
        }
        previous = std::move(current);
        previous_z = current_z;
        has_previous = true;
        current.clear();
        current_parts.clear();
        has_current = false;
    };

    for (size_t start = 0; start < by_z.size(); start += batch) {
        const size_t stop = std::min(by_z.size(), start + batch);
        std::vector<std::vector<Point>> read(stop - start);
        std::vector<Centre> centres(stop - start);
        std::vector<char> has_centre(stop - start, 0);
        parallel_for(static_cast<int>(stop - start), workers, [&](int i, int) {
            bool has = false;
            read[i] = read_file(files[by_z[start + i].second], z_min, z_max, &centres[i], &has);
            has_centre[i] = has;
        });
        for (size_t i = 0; i < read.size(); ++i) {
            const size_t rank = by_z[start + i].second;
            // the points of a file normally all carry the z of its name; any other z is its own slice
            std::map<int, std::vector<Point>> parts;
            for (const Point& p : read[i]) parts[p.z].push_back(p);
            if (parts.size() > 1 || (parts.size() == 1 && parts.begin()->first != by_z[start + i].first)) {
                // the name does not tell the slices of this file: fall back to everything in memory
                std::vector<Point> points = load_points(results_dir, z_min, z_max, threads);
                std::map<int, int> correction = align_labels(points);
                return survey_of(points, std::move(correction));
            }
            if (has_centre[i] && by_z[start + i].first >= z_min && by_z[start + i].first <= z_max)
                survey.centres[static_cast<int>(by_z[start + i].first)] = centres[i];
            if (parts.empty()) continue;
            const int z = parts.begin()->first;
            if (has_current && z != current_z) close_slice();
            if (!has_current) {
                has_current = true;
                current_z = z;
            }
            current_parts.emplace_back(rank, read[i].size());
            current.insert(current.end(), read[i].begin(), read[i].end());
            std::vector<Point>().swap(read[i]);
        }
    }
    close_slice();
    order_layers(survey, first);
    print_alignment(moved, unsure);
    return survey;
}

std::vector<Point> load_layer(const std::string& results_dir, int z_min, int z_max, int threads,
                              const LayerSurvey& survey, int layer, const std::vector<int>& tracks) {
    const std::vector<std::string> files = files_in_range(results_dir, z_min, z_max);
    const std::set<int> wanted(tracks.begin(), tracks.end());
    std::vector<std::vector<Point>> per_file(files.size());
    parallel_for(static_cast<int>(files.size()), threads > 0 ? threads : hardware_threads(), [&](int i, int) {
        std::vector<Point> kept;
        for (Point& p : read_file(files[i], z_min, z_max)) {
            const auto c = survey.correction.find(p.z);
            const int shift = c == survey.correction.end() ? 0 : c->second;
            if (p.winding - shift != layer) continue;
            if (!wanted.empty() && !wanted.count(p.track)) continue;
            p.winding -= shift;
            kept.push_back(p);
        }
        per_file[i] = std::move(kept);
    });
    std::vector<Point> points;
    for (auto& v : per_file) {
        points.insert(points.end(), v.begin(), v.end());
        std::vector<Point>().swap(v);
    }
    return points;
}

void for_each_layer_slice(const std::string& results_dir, int z_min, int z_max, int threads,
                          const LayerSurvey& survey, int layer, const std::vector<int>& tracks,
                          const std::function<void(int, const std::vector<Point>&)>& fn) {
    const std::vector<std::string> files = files_in_range(results_dir, z_min, z_max);
    // in Z order when the names tell it (name order otherwise), a batch of files read at a time
    std::vector<std::pair<long long, size_t>> order;
    for (size_t r = 0; r < files.size(); ++r) {
        long long z = 0;
        order.emplace_back(z_of_name(files[r], z) ? z : LLONG_MAX, r);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const std::set<int> wanted(tracks.begin(), tracks.end());
    const int workers = threads > 0 ? threads : hardware_threads();
    const size_t batch = static_cast<size_t>(std::max(1, 2 * workers));
    std::map<int, std::vector<Point>> pending;  // slices not given yet (a batch may end in the middle of one)
    auto flush = [&](long long below) {          // gives the slices whose z is below `below`
        while (!pending.empty() && pending.begin()->first < below) {
            fn(pending.begin()->first, pending.begin()->second);
            pending.erase(pending.begin());
        }
    };
    for (size_t start = 0; start < order.size(); start += batch) {
        const size_t stop = std::min(order.size(), start + batch);
        std::vector<std::vector<Point>> kept(stop - start);
        parallel_for(static_cast<int>(stop - start), workers, [&](int i, int) {
            for (Point& p : read_file(files[order[start + i].second], z_min, z_max)) {
                const auto c = survey.correction.find(p.z);
                const int shift = c == survey.correction.end() ? 0 : c->second;
                if (p.winding - shift != layer) continue;
                if (!wanted.empty() && !wanted.count(p.track)) continue;
                p.winding -= shift;
                kept[i].push_back(p);
            }
        });
        for (auto& part : kept)
            for (Point& p : part) pending[p.z].push_back(p);
        // the next files start at their own z: everything before it is complete
        flush(stop < order.size() ? order[stop].first : LLONG_MAX);
    }
    flush(LLONG_MAX);
}

}  // namespace pw
