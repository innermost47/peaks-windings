// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "zarr_reader.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    // A float32 array of shape (rows, cols, 3): the volume coordinates (x, y, z) of every pixel, NaN where unknown.
    struct Coords
    {
        int rows = 0, cols = 0;
        std::vector<float> data;

        const float *at(int r, int c) const { return &data[(static_cast<size_t>(r) * cols + c) * 3]; }
        bool known(int r, int c) const
        {
            const float *p = at(r, c);
            return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        }
    };

    Coords load_npy(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("cannot read " + path);
        char magic[8];
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
            throw std::runtime_error(path + " is not a .npy file");
        const int major = static_cast<unsigned char>(magic[6]);
        size_t header_len = 0;
        if (major == 1)
        {
            unsigned char b[2];
            f.read(reinterpret_cast<char *>(b), 2);
            header_len = b[0] | (static_cast<size_t>(b[1]) << 8);
        }
        else
        {
            unsigned char b[4];
            f.read(reinterpret_cast<char *>(b), 4);
            header_len = b[0] | (static_cast<size_t>(b[1]) << 8) | (static_cast<size_t>(b[2]) << 16) |
                         (static_cast<size_t>(b[3]) << 24);
        }
        std::string header(header_len, '\0');
        f.read(&header[0], static_cast<std::streamsize>(header_len));
        if (!f)
            throw std::runtime_error(path + ": truncated header");
        if (header.find("'<f4'") == std::string::npos)
            throw std::runtime_error(path + ": float32 data expected");
        if (header.find("'fortran_order': False") == std::string::npos)
            throw std::runtime_error(path + ": C-order data expected");
        const size_t key = header.find("'shape'");
        const size_t open = header.find('(', key);
        const size_t close = header.find(')', open);
        if (key == std::string::npos || open == std::string::npos || close == std::string::npos)
            throw std::runtime_error(path + ": no shape in the header");
        std::vector<long long> shape;
        std::stringstream ss(header.substr(open + 1, close - open - 1));
        std::string item;
        while (std::getline(ss, item, ','))
            if (item.find_first_not_of(" ") != std::string::npos)
                shape.push_back(std::stoll(item));
        if (shape.size() != 3 || shape[2] != 3)
            throw std::runtime_error(path + ": shape (rows, cols, 3) expected");
        Coords c;
        c.rows = static_cast<int>(shape[0]);
        c.cols = static_cast<int>(shape[1]);
        c.data.resize(static_cast<size_t>(c.rows) * c.cols * 3);
        f.read(reinterpret_cast<char *>(c.data.data()), static_cast<std::streamsize>(c.data.size() * sizeof(float)));
        if (!f)
            throw std::runtime_error(path + ": truncated data");
        return c;
    }

    // 8-bit grey with the 1 % .. 99 % range of the non-zero voxels (a masked volume is mostly zeros).
    cv::Mat to_8bit(const cv::Mat &plane)
    {
        cv::Mat f;
        plane.convertTo(f, CV_32F);
        std::vector<float> values;
        values.reserve(f.total());
        for (int r = 0; r < f.rows; ++r)
        {
            const float *p = f.ptr<float>(r);
            for (int c = 0; c < f.cols; ++c)
                if (p[c] != 0.f && std::isfinite(p[c]))
                    values.push_back(p[c]);
        }
        cv::Mat out;
        if (values.size() < 2)
        {
            f.convertTo(out, CV_8U);
            return out;
        }
        auto pct = [&](double q)
        {
            const size_t k = static_cast<size_t>(q * (values.size() - 1));
            std::nth_element(values.begin(), values.begin() + k, values.end());
            return values[k];
        };
        const double lo = pct(0.01), hi = pct(0.99);
        const double span = std::max(hi - lo, 1e-6);
        f.convertTo(out, CV_8U, 255.0 / span, -lo * 255.0 / span);
        return out;
    }

    struct Options
    {
        std::string coords;
        std::string scroll;
        std::string output; // empty: <folder of the coords>/sections/
        int sections = 8;
        std::vector<int> z_list;
        int margin = 150;
        int max_size = 2500;
        double jump = 5.0;
        int threads = 0;
        int connections = 32;
    };

    void usage()
    {
        std::cout << "Draws an unrolled surface back onto the raw CT slices, to check that it stays on one sheet.\n\n"
                     "  --coords FILE       the _coords.npy file written by unroll_layer (required)\n"
                     "  --scroll URL        the CT volume given to unroll_layer: path or URL of a zarr store (required)\n"
                     "  --sections N        slices drawn, evenly spaced over the image (default 8)\n"
                     "  --z Z1,Z2,...       draw these slices instead\n"
                     "  --output DIR        (default: <folder of the coords>/sections/)\n"
                     "  --margin PX         voxels of volume kept around the curve (default 150)\n"
                     "  --max-size PX       longest side of an output image (default 2500)\n"
                     "  --jump F            a step longer than F times the median step is marked as a jump (default 5)\n"
                     "  --threads N         worker threads (default 0: all)\n"
                     "  --connections N     parallel http downloads (default 32)\n";
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
            if (arg == "--coords")
                o.coords = value;
            else if (arg == "--scroll")
                o.scroll = value;
            else if (arg == "--output")
                o.output = value;
            else if (arg == "--sections")
                o.sections = std::max(1, std::stoi(value));
            else if (arg == "--z")
            {
                std::stringstream ss(value);
                std::string item;
                while (std::getline(ss, item, ','))
                    if (!item.empty())
                        o.z_list.push_back(std::stoi(item));
            }
            else if (arg == "--margin")
                o.margin = std::max(0, std::stoi(value));
            else if (arg == "--max-size")
                o.max_size = std::max(200, std::stoi(value));
            else if (arg == "--jump")
                o.jump = std::stod(value);
            else if (arg == "--threads")
                o.threads = std::stoi(value);
            else if (arg == "--connections")
                o.connections = std::stoi(value);
            else
                throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
        if (o.coords.empty() || o.scroll.empty())
            throw std::invalid_argument("--coords and --scroll are required (see --help)");
        if (o.output.empty())
        {
            const std::filesystem::path parent = std::filesystem::path(o.coords).parent_path();
            o.output = (parent / "sections").string();
        }
        return o;
    }

    // The slice of a row: the z of its first known pixel, or -1 when the row has fewer than two known pixels.
    int row_z(const Coords &c, int r)
    {
        int known = 0, z = -1;
        for (int k = 0; k < c.cols; ++k)
            if (c.known(r, k))
            {
                if (known++ == 0)
                    z = static_cast<int>(std::lround(c.at(r, k)[2]));
            }
        return known >= 2 ? z : -1;
    }

    // Draws one row of the coordinates on its slice. Returns the number of jumps found.
    int draw_section(const Coords &c, int r, int z, pw::ZarrVolume &volume, const Options &o, const std::string &path)
    {
        const std::vector<int64_t> &shape = volume.shape(); // (z, y, x)
        struct P
        {
            int k;
            double x, y;
        };
        std::vector<P> pts;
        double x_min = 1e300, x_max = -1e300, y_min = 1e300, y_max = -1e300;
        for (int k = 0; k < c.cols; ++k)
        {
            if (!c.known(r, k))
                continue;
            const float *p = c.at(r, k);
            pts.push_back({k, p[0], p[1]});
            x_min = std::min(x_min, static_cast<double>(p[0])), x_max = std::max(x_max, static_cast<double>(p[0]));
            y_min = std::min(y_min, static_cast<double>(p[1])), y_max = std::max(y_max, static_cast<double>(p[1]));
        }
        const int64_t x0 = std::max<int64_t>(0, static_cast<int64_t>(std::floor(x_min)) - o.margin);
        const int64_t y0 = std::max<int64_t>(0, static_cast<int64_t>(std::floor(y_min)) - o.margin);
        const int64_t x1 = std::min<int64_t>(shape[2], static_cast<int64_t>(std::ceil(x_max)) + o.margin + 1);
        const int64_t y1 = std::min<int64_t>(shape[1], static_cast<int64_t>(std::ceil(y_max)) + o.margin + 1);
        if (x1 <= x0 || y1 <= y0)
        {
            std::printf("    z=%d: the curve is outside the volume, skipped\n", z);
            return 0;
        }
        const std::vector<cv::Mat> region = volume.read_region(z, z + 1, y0, y1, x0, x1);
        if (region.empty() || region[0].empty())
        {
            std::printf("    z=%d: nothing read, skipped\n", z);
            return 0;
        }
        cv::Mat gray = to_8bit(region[0]), canvas;
        const double scale = std::min(1.0, o.max_size / static_cast<double>(std::max(gray.cols, gray.rows)));
        if (scale < 1.0)
            cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::cvtColor(gray, canvas, cv::COLOR_GRAY2BGR);

        // the median step between neighbouring columns, to tell a jump from an ordinary step
        std::vector<double> steps;
        for (size_t i = 1; i < pts.size(); ++i)
            if (pts[i].k == pts[i - 1].k + 1)
                steps.push_back(std::hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
        double median = 1.0;
        if (!steps.empty())
        {
            std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
            median = std::max(steps[steps.size() / 2], 1e-6);
        }
        const double limit = o.jump * median;

        auto to_pixel = [&](const P &p)
        {
            return cv::Point(static_cast<int>(std::lround((p.x - x0) * scale)),
                             static_cast<int>(std::lround((p.y - y0) * scale)));
        };
        const int thickness = std::max(2, static_cast<int>(std::lround(3 * scale)));
        int jumps = 0;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            // colour along the image: hue from red (first column) to violet (last column)
            const int hue = static_cast<int>(150.0 * pts[i].k / std::max(1, c.cols - 1));
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255)), bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            const cv::Vec3b v = bgr.at<cv::Vec3b>(0, 0);
            const cv::Scalar colour(v[0], v[1], v[2]);
            if (i == 0 || pts[i].k > pts[i - 1].k + 1)
            {
                cv::circle(canvas, to_pixel(pts[i]), thickness, colour, cv::FILLED, cv::LINE_AA);
                continue;
            }
            const double d = std::hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
            if (d > limit)
            {
                // two neighbouring columns far apart: the unrolling jumped
                ++jumps;
                cv::circle(canvas, to_pixel(pts[i - 1]), 4 * thickness, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
                cv::circle(canvas, to_pixel(pts[i]), 4 * thickness, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
                continue;
            }
            cv::line(canvas, to_pixel(pts[i - 1]), to_pixel(pts[i]), colour, thickness, cv::LINE_AA);
        }
        char text[160];
        std::snprintf(text, sizeof(text), "z = %d | %zu points | %d jump(s) | colour: first column red -> last violet", z,
                      pts.size(), jumps);
        const double font_scale = std::max(0.6, canvas.cols / 1800.0);
        cv::putText(canvas, text, cv::Point(15, static_cast<int>(40 * font_scale)), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
        cv::putText(canvas, text, cv::Point(15, static_cast<int>(40 * font_scale)), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        if (!cv::imwrite(path, canvas))
            throw std::runtime_error("cannot write " + path);
        std::printf("    z=%d: %zu points, %d jump(s) -> %s\n", z, pts.size(), jumps, path.c_str());
        std::fflush(stdout);
        return jumps;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options o = parse(argc, argv);
        const Coords c = load_npy(o.coords);
        std::printf("--> %s: %d rows, %d columns\n", o.coords.c_str(), c.rows, c.cols);

        const std::unique_ptr<pw::ZarrVolume> volume = pw::ZarrVolume::open(o.scroll, o.connections, o.threads);
        std::printf("--> %s\n", volume->description().c_str());
        const std::vector<int64_t> &shape = volume->shape();

        // the coordinates must be in voxels of this volume: a curve beyond its edges means another resolution level
        float max_x = 0, max_y = 0, max_z = 0;
        for (int r = 0; r < c.rows; ++r)
            for (int k = 0; k < c.cols; ++k)
                if (c.known(r, k))
                {
                    const float *p = c.at(r, k);
                    max_x = std::max(max_x, p[0]), max_y = std::max(max_y, p[1]), max_z = std::max(max_z, p[2]);
                }
        std::printf("--> Coordinates reach x=%.0f, y=%.0f, z=%.0f; the volume is x<%lld, y<%lld, z<%lld\n", max_x, max_y,
                    max_z, static_cast<long long>(shape[2]), static_cast<long long>(shape[1]),
                    static_cast<long long>(shape[0]));
        if (max_x >= shape[2] || max_y >= shape[1] || max_z >= shape[0])
            std::printf("[warning] the coordinates go beyond the volume: is it the resolution level the points were "
                        "computed on?\n");

        // the rows to draw
        std::vector<int> rows, zs;
        if (!o.z_list.empty())
        {
            for (int wanted : o.z_list)
            {
                int best = -1, best_d = 0;
                for (int r = 0; r < c.rows; ++r)
                {
                    const int z = row_z(c, r);
                    if (z < 0)
                        continue;
                    const int d = std::abs(z - wanted);
                    if (best < 0 || d < best_d)
                        best = r, best_d = d;
                }
                if (best < 0)
                    continue;
                if (best_d > 0)
                    std::printf("--> z=%d is not in the image: the nearest slice, z=%d, is drawn\n", wanted, row_z(c, best));
                rows.push_back(best);
            }
        }
        else
        {
            std::vector<int> usable;
            for (int r = 0; r < c.rows; ++r)
                if (row_z(c, r) >= 0)
                    usable.push_back(r);
            const int n = std::min<int>(o.sections, static_cast<int>(usable.size()));
            for (int i = 0; i < n; ++i)
                rows.push_back(usable[n == 1 ? usable.size() / 2 : static_cast<size_t>(i) * (usable.size() - 1) / (n - 1)]);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        if (rows.empty())
            throw std::runtime_error("no row of the coordinates has two known pixels");

        std::filesystem::create_directories(o.output);
        std::printf("--> Drawing %zu slice(s) to '%s'\n", rows.size(), o.output.c_str());
        int total_jumps = 0, drawn = 0;
        for (int r : rows)
        {
            const int z = row_z(c, r);
            if (z < 0 || z >= shape[0])
            {
                std::printf("    z=%d: outside the volume, skipped\n", z);
                continue;
            }
            char name[64];
            std::snprintf(name, sizeof(name), "section_z%05d.png", z);
            total_jumps += draw_section(c, r, z, *volume, o, (std::filesystem::path(o.output) / name).string());
            ++drawn;
        }
        std::printf("--> %d slice(s) drawn, %d jump(s) in all\n", drawn, total_jumps);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}