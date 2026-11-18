// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "parallel.hpp"
#include "scroll_settings.hpp"
#include "windings.hpp"
#include "zarr_reader.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    const double PI = 3.14159265358979323846;

    using Clock = std::chrono::steady_clock;

    double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

    struct Options
    {
        pw::Params params;
        std::string scroll_url;
        std::string ct; // CT volume of the same scroll, recorded for the next tools
        long long z_start = 0;
        long long z_count = 25000;
        long long every = 1;
        std::string solver = "lsq";
        std::string output_dir;
        std::string scroll_name = "scroll";
        std::string image;
        bool synthetic = false;
        double delamination = 25;
        int connections = 32;
        double cache_gb = 2.0;
        bool timings = false;
        bool frame_px_given = false;
        bool scale_given = false;
    };

    void usage()
    {
        std::cout << "Winding labels from several peaks, made consistent by one network.\n\n"
                     "  --scroll-url URL      zarr store: local directory, http(s):// or s3:// (anonymous); default: the\n"
                     "                        surface volume recorded in <output dir>/scroll_settings.json\n"
                     "  --ct URL              the CT volume of the same scroll, at the same resolution: recorded in\n"
                     "                        scroll_settings.json for unroll_layer and overlay_sections\n"
                     "  --z-start N           first slice (default 8000)\n"
                     "  --z-count N           number of slices (default 9000)\n"
                     "  --every N             process one slice in N (default 1)\n"
                     "  --scale F             downsampling for the mask and the relief only (default 0.3)\n"
                     "  --peaks N             summits kept (default 5)\n"
                     "  --rays N              rays of every summit, at fixed angles (default 360)\n"
                     "  --track-radius F      full-res px: a summit found this close to a tracked one is the same (default 30)\n"
                     "  --track-memory N      slices a tracked summit may stay unseen before its track ends (default 3)\n"
                     "  --max-radius F        0: the size of the slice (default 0)\n"
                     "  --min-spire N         px of spire for a crossing (default 2)\n"
                     "  --min-gap N           px of void before a new crossing (default 3)\n"
                     "  --frame-px F          full-res px: thicker than this is the frame (default 30)\n"
                     "  --close-px F          full-res px: gaps closed to find the interior (default 70)\n"
                     "  --output-dir DIR      (default results/json/<scroll name>/)\n"
                     "  --scroll-name NAME    name of the scroll, used in the file names and folders (default scroll)\n"
                     "  --image FILE          process one image file instead of a zarr store\n"
                     "  --synthetic           test on a synthetic spiral, checked against truth\n"
                     "  --delamination F      synthetic test: width of the delamination in px (default 25)\n"
                     "  --threads N           worker threads (default 0: all)\n"
                     "  --centre-jobs N       candidate centres tried at the same time (default 4)\n"
                     "  --connections N       parallel http downloads, or file reads of a local store (default 32)\n"
                     "  --cache-gb F          a chunk plane is kept decoded in memory up to this size (default 2)\n"
                     "  --timings             print the time of every step\n";
    }

    Options parse(int argc, char **argv)
    {
        Options o;
        std::map<std::string, std::function<void(const std::string &)>> with_value = {
            {"--scroll-url", [&](const std::string &v)
             { o.scroll_url = v; }},
            {"--ct", [&](const std::string &v)
             { o.ct = v; }},
            {"--z-start", [&](const std::string &v)
             { o.z_start = std::stoll(v); }},
            {"--z-count", [&](const std::string &v)
             { o.z_count = std::stoll(v); }},
            {"--every", [&](const std::string &v)
             { o.every = std::stoll(v); }},
            {"--scale", [&](const std::string &v)
             { o.params.scale = std::stod(v); o.scale_given = true; }},
            {"--peaks", [&](const std::string &v)
             { o.params.peaks = std::stoi(v); }},
            {"--rays", [&](const std::string &v)
             { o.params.rays = std::stoi(v); }},
            {"--track-radius", [&](const std::string &v)
             { o.params.track_radius = std::stod(v); }},
            {"--track-memory", [&](const std::string &v)
             { o.params.track_memory = std::stoi(v); }},
            {"--max-radius", [&](const std::string &v)
             { o.params.max_radius = std::stod(v); }},
            {"--min-spire", [&](const std::string &v)
             { o.params.min_spire = std::stoi(v); }},
            {"--min-gap", [&](const std::string &v)
             { o.params.min_gap = std::stoi(v); }},
            {"--frame-px", [&](const std::string &v)
             { o.params.frame_px = std::stod(v); o.frame_px_given = true; }},
            {"--close-px", [&](const std::string &v)
             { o.params.close_px = std::stod(v); }},
            {"--solver", [&](const std::string &v)
             { o.solver = v; }},
            {"--output-dir", [&](const std::string &v)
             { o.output_dir = v; }},
            {"--scroll-name", [&](const std::string &v)
             { o.scroll_name = v; }},
            {"--image", [&](const std::string &v)
             { o.image = v; }},
            {"--delamination", [&](const std::string &v)
             { o.delamination = std::stod(v); }},
            {"--threads", [&](const std::string &v)
             { o.params.threads = std::stoi(v); }},
            {"--centre-jobs", [&](const std::string &v)
             { o.params.centre_jobs = std::stoi(v); }},
            {"--connections", [&](const std::string &v)
             { o.connections = std::stoi(v); }},
            {"--cache-gb", [&](const std::string &v)
             { o.cache_gb = std::stod(v); }},
        };
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                usage();
                std::exit(0);
            }
            if (arg == "--synthetic")
            {
                o.synthetic = true;
                continue;
            }
            if (arg == "--timings")
            {
                o.timings = true;
                continue;
            }
            std::string value;
            const size_t eq = arg.find('=');
            bool has_value = false;
            if (eq != std::string::npos)
            {
                value = arg.substr(eq + 1);
                arg = arg.substr(0, eq);
                has_value = true;
            }
            const auto it = with_value.find(arg);
            if (it == with_value.end())
                throw std::invalid_argument("unknown option " + arg + " (see --help)");
            if (!has_value)
            {
                if (i + 1 >= argc)
                    throw std::invalid_argument("option " + arg + " needs a value");
                value = argv[++i];
            }
            try
            {
                it->second(value);
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("invalid value '" + value + "' for " + arg);
            }
        }
        if (o.every < 1)
            throw std::invalid_argument("--every must be at least 1");
        if (o.params.scale <= 0)
            throw std::invalid_argument("--scale must be positive");
        if (o.scroll_name.empty())
            throw std::invalid_argument("--scroll-name cannot be empty");
        if (o.output_dir.empty())
            o.output_dir = (std::filesystem::path("results") / "json" / o.scroll_name).string();
        return o;
    }

    std::string centre_text(const pw::SliceResult &r)
    {
        if (!r.has_centre)
            return "none";
        char buf[96];
        std::snprintf(buf, sizeof(buf), "(%.1f, %.1f) peak %d track %d", r.cx, r.cy, r.centre_peak, r.centre_track);
        return buf;
    }

    void print_timings(const pw::Timings &t)
    {
        std::printf("    mask %.2f s, peaks %.2f s, rays %.2f s, skeleton %.2f s, centre %.2f s, output %.2f s\n", t.mask,
                    t.peaks, t.rays, t.skeleton, t.centre, t.output);
    }

    void write_file(const std::string &path, const std::string &text)
    {
        std::FILE *f = std::fopen(path.c_str(), "wb");
        if (!f)
            throw std::runtime_error("cannot write " + path);
        const size_t written = std::fwrite(text.data(), 1, text.size(), f);
        if (std::fclose(f) != 0 || written != text.size())
            throw std::runtime_error("cannot write " + path);
    }

    // ------------------------------------------------------------------------------------------- synthetic test
    // A spiral scroll in its case, with breaks, noise and a delamination; and the points of its true centre line.
    struct Synthetic
    {
        cv::Mat img;
        std::vector<cv::Point2d> pts;
        std::vector<double> theta;
        double cx = 0, cy = 0;

        // Turn number of the sheet nearest to (x, y), counted from the inside (1 = innermost).
        int truth(double x, double y) const
        {
            size_t best = 0;
            double best_d = 1e300;
            for (size_t i = 0; i < pts.size(); ++i)
            {
                const double dx = pts[i].x - x, dy = pts[i].y - y;
                const double d = dx * dx + dy * dy;
                if (d < best_d)
                {
                    best_d = d;
                    best = i;
                }
            }
            return static_cast<int>(std::floor(theta[best] / (2 * PI))) + 1;
        }
    };

    Synthetic synthetic_slice(double delamination, int size = 1200, int turns = 12, double pitch = 40, double r0 = 30,
                              int thickness = 6, unsigned seed = 0)
    {
        Synthetic s;
        std::mt19937_64 rng(seed);
        s.cx = size / 2.0;
        s.cy = size / 2.0 + 20;
        cv::Mat img(size, size, CV_8U, cv::Scalar(30));
        cv::circle(img, cv::Point(static_cast<int>(s.cx), static_cast<int>(s.cy)),
                   static_cast<int>(r0 + pitch * turns + 40), cv::Scalar(200), 14); // the case
        const int n = 20000;
        std::vector<double> gap(n), prefix(n + 1, 0.0);
        s.theta.resize(n);
        for (int i = 0; i < n; ++i)
        {
            s.theta[i] = 2 * PI * turns * i / (n - 1);
            const double phase = std::fmod(s.theta[i], 2 * PI);
            gap[i] = (s.theta[i] > 2 * PI * 6.1 && phase > 1.0 && phase < 2.2) ? delamination : 0.0;
            prefix[i + 1] = prefix[i] + gap[i];
        }
        s.pts.resize(n);
        for (int i = 0; i < n; ++i)
        {
            // np.convolve(gap, ones(400) / 400, mode="same"): the mean of gap[i - 200 .. i + 199]
            const int lo = std::max(i - 200, 0), hi = std::min(i + 199, n - 1);
            const double smooth = (prefix[hi + 1] - prefix[lo]) / 400.0;
            const double r = r0 + pitch * s.theta[i] / (2 * PI) + smooth;
            s.pts[i] = {s.cx + r * std::cos(s.theta[i]), s.cy + r * std::sin(s.theta[i])};
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        auto to_pixel = [](const cv::Point2d &p)
        {
            return cv::Point(static_cast<int>(std::nearbyint(p.x)), static_cast<int>(std::nearbyint(p.y)));
        };
        for (int i = 0; i + 1 < n; ++i)
        {
            if (uniform(rng) < 0.0015)
                continue; // breaks in the sheet
            cv::line(img, to_pixel(s.pts[i]), to_pixel(s.pts[i + 1]), cv::Scalar(200), thickness);
        }
        std::normal_distribution<double> noise(0.0, 18.0);
        for (int r = 0; r < size; ++r)
        {
            unsigned char *row = img.ptr<unsigned char>(r);
            for (int c = 0; c < size; ++c)
                row[c] = static_cast<unsigned char>(std::min(std::max(row[c] + noise(rng), 0.0), 255.0));
        }
        s.img = img;
        return s;
    }

    void run_synthetic(Options &o)
    {
        if (!o.frame_px_given)
            o.params.frame_px = 10; // the synthetic case is 14 px thick, its sheets 6 px
        if (!o.scale_given)
            o.params.scale = 0.5; // at 0.3 its 6 px sheets and 14 px case are too close to tell apart
        const Synthetic s = synthetic_slice(o.delamination);
        const auto t0 = Clock::now();
        const pw::SliceResult info = pw::process_slice(s.img, o.params, nullptr);
        const double dt = seconds_since(t0);
        std::printf("true centre (%.0f, %.0f); chosen centre %s  (%.2f s, solver lsq)\n", s.cx, s.cy,
                    centre_text(info).c_str(), dt);
        if (o.timings)
            print_timings(info.timings);
        for (size_t k = 0; k < info.peaks.size(); ++k)
            std::printf("  peak %zu (track %d): (%.0f, %.0f) height %.1f score %.3f\n", k, info.peaks[k].track,
                        info.peaks[k].x, info.peaks[k].y, info.peaks[k].height, info.scores[k]);
        long right = 0, total = 0, raw_right = 0, high_pull = 0, high_pull_wrong = 0;
        for (const pw::LineOut &line : info.lines)
        {
            int k = 0;
            for (const pw::CrossingOut &c : line.crossings)
            {
                ++k;
                const int t = s.truth(c.x, c.y);
                ++total;
                right += c.winding_label == t;
                raw_right += k == t; // the old way: the count along the ray from its own peak
                if (c.pull > 3)
                {
                    ++high_pull;
                    high_pull_wrong += c.winding_label != t;
                }
            }
        }
        std::printf("crossings: %ld\n", total);
        if (total == 0)
            return;
        std::printf("  old labels (count from each ray's own peak): %6.1f%% right\n", 100.0 * raw_right / total);
        std::printf("  network labels (all peaks, one network):     %6.1f%% right\n", 100.0 * right / total);
        std::printf("  crossings with a pull above 3 sigma: %ld (of which wrong: %ld)\n", high_pull, high_pull_wrong);
    }

    // ------------------------------------------------------------------------------------------- one image file
    void run_image(const Options &o)
    {
        cv::Mat img = cv::imread(o.image, cv::IMREAD_UNCHANGED);
        if (img.empty())
            throw std::runtime_error("cannot read image " + o.image);
        if (img.channels() == 4)
            cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
        if (img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        if (img.depth() != CV_8U)
            img = pw::normalise(img); // an 8-bit image is taken as it is
        const auto t0 = Clock::now();
        const pw::SliceResult info = pw::process_slice(img, o.params, nullptr);
        const double dt = seconds_since(t0);
        std::filesystem::create_directories(o.output_dir);
        const std::string stem = std::filesystem::path(o.image).stem().string();
        const std::string path = (std::filesystem::path(o.output_dir) / (o.scroll_name + "_" + stem + ".json")).string();
        write_file(path, pw::to_json(info, -1));
        size_t n = 0;
        for (const pw::LineOut &line : info.lines)
            n += line.crossings.size();
        std::printf("--> %s: %zu crossings, centre %s, %.2f s -> %s\n", o.image.c_str(), n, centre_text(info).c_str(), dt,
                    path.c_str());
        if (o.timings)
            print_timings(info.timings);
    }

    // ------------------------------------------------------------------------------------------- a zarr volume
    // The surface volume read (and the CT volume, when given) recorded next to the results. Results already made from
    // a volume of another shape are in other voxels: they are not mixed.
    void record_volumes(pw::ScrollSettings &settings, const Options &o, const std::string &store,
                        const pw::ZarrVolume &volume)
    {
        const std::vector<int64_t> shape(volume.shape().begin(), volume.shape().end());
        if (!settings.surface.shape.empty() && settings.surface.shape != shape)
            throw std::runtime_error("the results in " + o.output_dir + " were made from a volume of shape " +
                                     pw::shape_text(settings.surface.shape) + ", this one is " +
                                     pw::shape_text(shape) + ": give another --scroll-name (or --output-dir)");
        settings.surface = {store, shape};
        if (!o.ct.empty())
            settings.ct = {o.ct, {}};
        settings.save();
        std::printf("--> volumes recorded in %s\n", settings.file().c_str());
    }

    void run_volume(const Options &o)
    {
        std::filesystem::create_directories(o.output_dir);
        pw::ScrollSettings settings = pw::ScrollSettings::load(o.output_dir);
        const std::string store = !o.scroll_url.empty() ? o.scroll_url : settings.surface.store;
        if (store.empty())
            throw std::invalid_argument("--scroll-url is required: no surface volume is recorded in " + settings.file());
        std::printf("--> zarr store: %s\n", store.c_str());
        std::fflush(stdout);
        const std::unique_ptr<pw::ZarrVolume> volume = pw::ZarrVolume::open(store, o.connections, o.params.threads, o.cache_gb);
        std::printf("--> %s\n", volume->description().c_str());
        record_volumes(settings, o, store, *volume);
        std::fflush(stdout);

        bool has_previous = false;
        cv::Point2d previous;
        pw::TrackState tracks; // the summits followed from slice to slice
        pw::SpotHeights spots;  // the solved sheet points of the previous slice
        const auto t_all = Clock::now();
        const long long z_end = std::min<long long>(o.z_start + o.z_count, volume->shape()[0]);
        if (o.z_start + o.z_count > volume->shape()[0])
            std::printf("--> the volume has %lld slices: stopping at z=%lld\n",
                        static_cast<long long>(volume->shape()[0]), z_end - 1);
        for (long long z = o.z_start; z < z_end; z += o.every)
        {
            auto t0 = Clock::now();
            const cv::Mat img = pw::normalise(volume->read_slice(z));
            const double t_read = seconds_since(t0);
            t0 = Clock::now();
            pw::SliceResult info = pw::process_slice(img, o.params, has_previous ? &previous : nullptr, &tracks, &spots);
            has_previous = info.has_centre;
            previous = cv::Point2d(info.cx, info.cy);
            spots = std::move(info.spots);
            size_t n = 0;
            for (const pw::LineOut &line : info.lines)
                n += line.crossings.size();
            char name[64];
            std::snprintf(name, sizeof(name), "_z%05lld.json", z);
            const std::string path = (std::filesystem::path(o.output_dir) / (o.scroll_name + name)).string();
            write_file(path, pw::to_json(info, z));
            const double t_compute = seconds_since(t0);
            std::string track_list;
            for (const pw::Peak &peak : info.peaks)
                track_list += (track_list.empty() ? "" : " ") + std::to_string(peak.track);
            std::printf("--> z=%lld: %zu crossings, centre %s, tracks [%s], %.2f s (read %.2f s, compute %.2f s) -> %s\n",
                        z, n, centre_text(info).c_str(), track_list.c_str(), t_read + t_compute, t_read, t_compute,
                        path.c_str());
            if (o.timings)
                print_timings(info.timings);
            std::fflush(stdout);
        }
        std::printf("done in %.1f s\n", seconds_since(t_all));
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        Options o = parse(argc, argv);
        if (o.synthetic)
            run_synthetic(o);
        else if (!o.image.empty())
            run_image(o);
        else
            run_volume(o);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}