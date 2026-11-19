// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "layer_unfolding.hpp"
#include "layers.hpp"
#include "parallel.hpp"
#include "scroll_settings.hpp"
#include "slab_sampler.hpp"
#include "unroll_output.hpp"
#include "zarr_reader.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

    struct Options
    {
        std::string scroll_name = "scroll"; // folders: results/{json,png}/<scroll name>/
        std::string results_dir;            // empty: results/json/<scroll name>
        std::string layer_spec;       // empty: the densest layer; "all"; or "N[,M...]"
        size_t min_layer_points = 1000; // --layer all: smaller layers are not unrolled
        std::string scroll;
        int step_z = 1;
        std::string output; // empty: results/png/<scroll name>/layer_<N>/unrolled_layer_<N>.png
        int connections = 32;
        int z_min = INT_MIN, z_max = INT_MAX;
        int max_width = 1600; // widest image along the sheet (px)
        bool diagnose_only = false; // follow the sheet and report, without reading the CT volume
        pw::SlabParams slab;
        pw::LayerUnfolding::Params unfolding;
    };

    // The layers to unroll, from the counts of the survey: the densest one (no --layer), the ones listed, or with
    // --layer all every layer of at least --min-points points, the densest first.
    std::vector<int> choose_layers(const pw::LayerSurvey &survey, const Options &o)
    {
        if (survey.layer_order.empty())
            throw std::runtime_error("the result files hold no point");
        std::vector<int> by_size = survey.layer_order;
        std::stable_sort(by_size.begin(), by_size.end(), [&](int a, int b)
                         { return survey.layer_points.at(a) > survey.layer_points.at(b); });
        std::vector<int> chosen;
        if (o.layer_spec.empty())
            chosen.push_back(by_size.front());
        else if (o.layer_spec == "all")
        {
            for (int w : by_size)
                if (survey.layer_points.at(w) >= o.min_layer_points)
                    chosen.push_back(w);
            if (chosen.empty())
                throw std::runtime_error("no layer has " + std::to_string(o.min_layer_points) +
                                         " points or more (--min-points)");
        }
        else
        {
            std::stringstream list(o.layer_spec);
            std::string item;
            while (std::getline(list, item, ','))
            {
                if (item.empty())
                    continue;
                size_t used = 0;
                const int w = std::stoi(item, &used);
                if (used != item.size())
                    throw std::invalid_argument("--layer: '" + item + "' is not a layer number");
                if (!survey.layer_points.count(w))
                    throw std::runtime_error("layer " + std::to_string(w) + " has no point");
                if (std::find(chosen.begin(), chosen.end(), w) == chosen.end())
                    chosen.push_back(w);
            }
            if (chosen.empty())
                throw std::invalid_argument("--layer: no layer given");
        }
        std::printf("--> Layer(s) to unroll: %zu of %zu (", chosen.size(), by_size.size());
        for (size_t i = 0; i < chosen.size(); ++i)
            std::printf("%s%d", i ? ", " : "", chosen[i]);
        std::printf(")\n");
        return chosen;
    }

    // The points of the result files are in voxels of the volume peaks_windings read: the volume sampled must be at the
    // same resolution (the same level of a multiscale store).
    void check_resolution(const pw::LayerUnfolding &unfolding, const pw::ZarrVolume &volume)
    {
        const auto &shape = volume.shape(); // (z, y, x)
        const double depth = static_cast<double>(shape[0]), height = static_cast<double>(shape[1]),
                     width = static_cast<double>(shape[2]);
        char where[256];
        std::snprintf(where, sizeof(where),
                      "the sheet reaches x=%.0f, y=%.0f, z=%d, the volume is (z, y, x) = (%.0f, %.0f, %.0f)",
                      unfolding.x_max(), unfolding.y_max(), unfolding.z_last(), depth, height, width);
        if (unfolding.x_max() >= width || unfolding.y_max() >= height || unfolding.z_last() >= depth)
            throw std::runtime_error(std::string(where) +
                                     ": it is not at the resolution peaks_windings read. Give --scroll the level of "
                                     "the store with the same shape as the one peaks_windings printed (URL/0, URL/1, "
                                     "URL/2...).");
        if (unfolding.x_max() < 0.5 * width && unfolding.y_max() < 0.5 * height)
            std::printf("--> warning: %s; the sheet lies in a corner of the volume: is the volume finer than the one "
                        "peaks_windings read? (compare the shapes both tools print)\n",
                        where);
    }

    // The CT volume: the one given, or else the one recorded for the scroll, at the shape of the surface volume
    // peaks_windings read (the voxels of the result files): another level of the same multiscale store is taken when
    // the one given has another shape. It is set in `settings`, for the caller to save.
    std::unique_ptr<pw::ZarrVolume> open_ct(const Options &o, pw::ScrollSettings &settings)
    {
        const std::string store = !o.scroll.empty() ? o.scroll : settings.ct.store;
        std::printf("--> CT volume: %s%s\n", store.c_str(), o.scroll.empty() ? " (recorded)" : "");
        std::string level;
        std::unique_ptr<pw::ZarrVolume> volume =
            pw::ZarrVolume::open_level(store, settings.surface.shape, level, o.connections, o.slab.threads);
        if (level != store)
            std::printf("--> level with the surface shape %s: %s\n", pw::shape_text(settings.surface.shape).c_str(),
                        level.c_str());
        std::printf("--> %s\n", volume->description().c_str());
        const std::vector<int64_t> shape(volume->shape().begin(), volume->shape().end());
        settings.ct = {level, shape}; // saved once the sheet is known to lie inside it
        return volume;
    }

    // Whether any pixel was sampled (its coordinates are finite).
    bool sampled_anything(const cv::Mat &coords)
    {
        for (int r = 0; r < coords.rows; ++r)
        {
            const cv::Vec3f *row = coords.ptr<cv::Vec3f>(r);
            for (int c = 0; c < coords.cols; ++c)
                if (std::isfinite(row[c][0]))
                    return true;
        }
        return false;
    }

    void usage()
    {
        std::cout << "Unroll one winding layer into a flat image (slab MIP), following its sheet slice after slice.\n\n"
                     "  --scroll-name NAME  name of the dataset, the same as given to peaks_windings (default: scroll);\n"
                     "                      results are read from results/json/NAME/ and written to\n"
                     "                      results/png/NAME/layer_<N>/\n"
                     "  --results-dir DIR   folder containing the JSON results (default: results/json/NAME)\n"
                     "  --layer N[,M...]    winding layer(s) to unroll, one image each (default: the one with the most\n"
                     "                      points); \"all\": every layer of at least --min-points points, densest first\n"
                     "  --min-points N      --layer all: smaller layers are not unrolled (default: 1000)\n"
                     "  --scroll URL        CT volume to sample: path or URL of a zarr store (default: the one recorded\n"
                     "                      in <results dir>/scroll_settings.json; the one given is recorded there)\n"
                     "  --slab N            slab thickness in voxels for the MIP (default: 6)\n"
                     "  --step-z N          vertical step in voxels (default: 1)\n"
                     "  --output FILE       output image (default: results/png/NAME/layer_<N>/unrolled_layer_<N>.png)\n"
                     "  --max-pull F        traces pulled further than F sigma by the labelling are left out\n"
                     "                      (default: 3; 0: none is)\n"
                     "  --max-gap F         voxels: a trace goes on from the end of another across at most this gap\n"
                     "                      (default: 60)\n"
                     "  --max-turn F        degrees: ... and turning at most this much (default: 45)\n"
                     "  --fill-px F         voxels: holes along the sheet up to this long are bridged straight\n"
                     "                      (default: 30)\n"
                     "  --fill-z N          slices: a column unknown in a slice takes the sheet of the nearest slice, at\n"
                     "                      most N away, that knows it (default: 0, never)\n"
                     "  --memory-slices N   slices looked back at to find the sheet of a trace: a region survives a\n"
                     "                      trace missing for up to N-1 slices (default: 1, the previous slice only)\n"
                     "  --inherit-radius F  voxels: a point finds the sheet of an earlier slice within this distance\n"
                     "                      (default: 4)\n"
                     "  --min-region-share F regions smaller than this share of the largest (by area) are left out\n"
                     "                      (default: 0.02)\n"
                     "  --max-width N       widest image along the sheet in px (default: 1600)\n"
                     "  --diagnose-only     follow the sheet and print the report (traces left out, regions, coverage),\n"
                     "                      without reading the CT volume or writing any image\n"
                     "  --cache-mb F        memory budget of the decoded volume chunks held at once, in MB\n"
                     "                      (default: 1024); only the chunks the slab crosses are read\n"
                     "  --z-min N           first slice kept (default: all)\n"
                     "  --z-max N           last slice kept (default: all)\n"
                     "  --threads N         worker threads (default 0: all); --workers is the same option\n"
                     "  --connections N     parallel http downloads, or file reads of a local store (default 32)\n";
    }

    Options parse(int argc, char **argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                usage();
                std::exit(0);
            }
            if (arg == "--diagnose-only")
            {
                o.diagnose_only = true;
                continue;
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
            else if (arg == "--layer" || arg == "--layers")
                o.layer_spec = value;
            else if (arg == "--min-points")
                o.min_layer_points = static_cast<size_t>(std::max(0, std::stoi(value)));
            else if (arg == "--scroll")
                o.scroll = value;
            else if (arg == "--slab")
                o.slab.slab = std::stoi(value);
            else if (arg == "--step-z")
                o.step_z = std::stoi(value);
            else if (arg == "--output")
                o.output = value;
            else if (arg == "--max-pull")
                o.unfolding.max_pull = std::stod(value);
            else if (arg == "--max-gap")
                o.unfolding.max_gap_px = std::stod(value);
            else if (arg == "--max-turn")
                o.unfolding.max_turn_deg = std::stod(value);
            else if (arg == "--fill-px")
                o.unfolding.fill_px = std::stod(value);
            else if (arg == "--fill-z")
                o.unfolding.fill_z = std::max(0, std::stoi(value));
            else if (arg == "--memory-slices")
                o.unfolding.memory_slices = std::max(1, std::stoi(value));
            else if (arg == "--inherit-radius")
                o.unfolding.inherit_radius_px = std::stod(value);
            else if (arg == "--min-region-share")
                o.unfolding.min_region_share = std::stod(value);
            else if (arg == "--max-width")
                o.max_width = std::max(300, std::stoi(value));
            else if (arg == "--cache-mb")
                o.slab.cache_mb = std::stod(value);
            else if (arg == "--z-min")
                o.z_min = std::stoi(value);
            else if (arg == "--z-max")
                o.z_max = std::stoi(value);
            else if (arg == "--threads" || arg == "--workers")
                o.slab.threads = std::stoi(value);
            else if (arg == "--connections")
                o.connections = std::stoi(value);
            else
                throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
        if (o.step_z < 1)
            throw std::invalid_argument("--step-z must be at least 1");
        if (o.slab.slab < 0)
            throw std::invalid_argument("--slab cannot be negative");
        if (!(o.slab.cache_mb > 0))
            throw std::invalid_argument("--cache-mb must be positive");
        const bool is_number = !o.scroll.empty() && o.scroll.find_first_not_of("0123456789") == std::string::npos;
        if (is_number)
            throw std::invalid_argument("--scroll must be the path or URL of the zarr CT volume, not a number");
        if (o.scroll_name.empty())
            throw std::invalid_argument("--scroll-name cannot be empty");
        if (o.results_dir.empty())
            o.results_dir = (std::filesystem::path("results") / "json" / o.scroll_name).string();
        return o;
    }
    // Unrolls one layer: its sheet followed slice after slice, then the CT volume sampled along it (opened at the first
    // layer that needs it). Returns whether an image was written; a layer without anything to unroll is skipped.
    bool unroll_one(const Options &o, const pw::LayerSurvey &survey, int layer, pw::ScrollSettings &settings,
                    std::unique_ptr<pw::ZarrVolume> &volume)
    {
        // the position along the sheet of every point of its traces
        pw::LayerUnfolding unfolding(o.unfolding);
        pw::for_each_layer_traces(o.results_dir, o.z_min, o.z_max, o.slab.threads, survey, layer,
                                  [&](int z, const std::vector<pw::Trace> &traces)
                                  { unfolding.add_slice(z, traces); });
        if (unfolding.empty())
        {
            std::printf("--> Layer %d has no trace of its sheet (result files written before the traces existed?): "
                        "skipped\n",
                        layer);
            return false;
        }
        unfolding.lay_out();
        const pw::LayerUnfolding::Stats &st = unfolding.stats();
        std::printf("--> Sheet followed over %ld slice(s): %ld trace(s) inherited from the slice before, %ld joined "
                    "across a junction or a break, %ld seeded a region, %ld left out; %ld region(s) laid side by side\n",
                    st.slices, st.inherited, st.joined, st.seeded, st.dropped, st.regions_kept);

        // the columns: evenly spaced positions along the sheet
        const double span = unfolding.width();
        const int width = std::max(2, std::min(o.max_width, static_cast<int>(std::ceil(span)) + 1));
        std::vector<double> u(width);
        for (int k = 0; k < width; ++k)
            u[k] = span * k / (width - 1);
        std::printf("    - Traces offered: %ld; left out: %ld pulled too far (--max-pull), %ld too short, "
                    "%ld not placed; regions: %ld made, %ld kept\n",
                    st.offered, st.pulled, st.too_short, st.unplaced, st.regions_total, st.regions_kept);
        std::printf("    - Coverage: %.1f%% of the columns have a sheet position in an average slice",
                    100.0 * unfolding.coverage(u, false));
        if (o.unfolding.fill_z > 0)
            std::printf(", %.1f%% once the nearest slices (--fill-z %d) fill the holes",
                        100.0 * unfolding.coverage(u, true), o.unfolding.fill_z);
        std::printf(" (the rest stays black)\n");
        const double total_area = unfolding.total_area();
        std::printf("    - Regions kept hold %.1f%% of the trace length placed; the largest ones (u extent, slices, share):\n",
                    total_area > 0 ? 100.0 * unfolding.kept_area() / total_area : 0.0);
        for (const pw::LayerUnfolding::RegionSummary &r : unfolding.largest_regions(10))
            std::printf("        %s width %9.0f voxels, Z %d -> %d, %.1f%% of the length\n", r.kept ? "kept   " : "left out",
                        r.width, r.z_first, r.z_last, total_area > 0 ? 100.0 * r.area / total_area : 0.0);
        const std::string name = "unrolled_layer_" + std::to_string(layer) + ".png";
        const std::string output =
            !o.output.empty()
                ? o.output
                : (std::filesystem::path("results") / "png" / o.scroll_name / ("layer_" + std::to_string(layer)) / name)
                      .string();
        std::vector<int> dense_z;
        for (int z = unfolding.z_first(); z <= unfolding.z_last(); z += o.step_z)
            dense_z.push_back(z);
        if (o.diagnose_only)
        {
            // what the image would show: white where the slice knows the sheet, grey where a near slice fills in
            cv::Mat map(static_cast<int>(dense_z.size()), width, CV_8U, cv::Scalar(0));
            pw::parallel_for(map.rows, o.slab.threads > 0 ? o.slab.threads : pw::hardware_threads(), [&](int r, int)
                             {
                const pw::Rail own = unfolding.rail(dense_z[r], u, false), filled = unfolding.rail(dense_z[r], u, true);
                unsigned char *row = map.ptr<unsigned char>(r);
                for (int c = 0; c < width; ++c)
                    row[c] = std::isfinite(own.xs[c]) ? 255 : std::isfinite(filled.xs[c]) ? 128 : 0; });
            std::printf("--> Layer %d: diagnosis only, the CT volume is not read; coverage map: '%s'\n", layer,
                        pw::save_coverage_map(map, output).c_str());
            return false;
        }
        std::printf("--> Unrolling:\n");
        std::printf("    - Width: %d pixels along the sheet (%.2f voxels per pixel)\n", width, span / (width - 1));
        std::printf("    - Height: %zu CT slices (Z=%d to %d)\n", dense_z.size(), dense_z.front(), dense_z.back());
        std::printf("    - Thickness: %d voxels in Maximum Intensity Projection (MIP)\n", o.slab.slab);

        // sampling, and the outputs
        if (!volume)
            volume = open_ct(o, settings);
        check_resolution(unfolding, *volume);
        settings.save();
        const pw::SlabImage sampled = pw::sample_slab(
            dense_z, width, [&](int z)
            { return unfolding.rail(z, u); },
            o.slab, *volume);
        if (!sampled_anything(sampled.coords))
        {
            std::printf("--> Layer %d: nothing could be sampled along the sheet: no image written\n", layer);
            return false;
        }

        pw::UnrolledImage image;
        image.values = sampled.values;
        image.coords = sampled.coords;
        image.z_start = dense_z.front();
        image.z_end = dense_z.back();
        image.title = "Unrolled layer " + std::to_string(layer) + " (slab MIP)";
        image.x_label = "Position along the sheet (voxels)";
        image.x_max = span;
        pw::save_unrolled(image, output);
        return true;
    }
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options o = parse(argc, argv);
        const auto t0 = Clock::now();
        pw::ScrollSettings settings = pw::ScrollSettings::load(o.results_dir);
        if (!o.diagnose_only && o.scroll.empty() && !settings.ct.known())
            throw std::invalid_argument("--scroll is required: no CT volume is recorded in " + settings.file() +
                                        " (give it once, here or to peaks_windings --ct)");

        // 1. The layers: the labels aligned along Z (two slices held at a time), their points counted
        const pw::LayerSurvey survey = pw::survey_layers(o.results_dir, o.z_min, o.z_max, o.slab.threads);
        const std::vector<int> layers = choose_layers(survey, o);
        if (!o.output.empty() && layers.size() > 1)
            throw std::invalid_argument("--output names one image: it cannot be used with several layers");

        // 2. Every layer in turn, the CT volume opened once
        std::unique_ptr<pw::ZarrVolume> volume;
        size_t written = 0;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            std::printf("--> Layer %d (%zu/%zu): %zu point(s)\n", layers[i], i + 1, layers.size(),
                        survey.layer_points.at(layers[i]));
            std::fflush(stdout);
            written += unroll_one(o, survey, layers[i], settings, volume);
        }
        if (o.diagnose_only)
            std::printf("--> %zu layer(s) diagnosed in %.1f s (diagnosis only: no image written).\n", layers.size(),
                        seconds_since(t0));
        else
            std::printf("--> %zu of %zu layer(s) unrolled in %.1f s.\n", written, layers.size(), seconds_since(t0));
        return written > 0 || o.diagnose_only ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
