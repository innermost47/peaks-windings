// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "unroll_output.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace pw
{
    namespace
    {
        double percentile(std::vector<double> &v, double q)
        {
            const double pos = (v.size() - 1) * q / 100.0;
            const size_t lo = static_cast<size_t>(std::floor(pos));
            std::nth_element(v.begin(), v.begin() + lo, v.end());
            const double a = v[lo];
            const double b = lo + 1 < v.size() ? *std::min_element(v.begin() + lo + 1, v.end()) : a;
            const double t = pos - lo;
            return t >= 0.5 ? b - (b - a) * (1.0 - t) : a + (b - a) * t;
        }

        inline void write_u16(std::ostream &os, uint16_t v)
        {
            const unsigned char buf[2] = {static_cast<unsigned char>(v & 0xff), static_cast<unsigned char>((v >> 8) & 0xff)};
            os.write(reinterpret_cast<const char *>(buf), 2);
        }

        inline void write_u32(std::ostream &os, uint32_t v)
        {
            const unsigned char buf[4] = {static_cast<unsigned char>(v & 0xff),
                                          static_cast<unsigned char>((v >> 8) & 0xff),
                                          static_cast<unsigned char>((v >> 16) & 0xff),
                                          static_cast<unsigned char>((v >> 24) & 0xff)};
            os.write(reinterpret_cast<const char *>(buf), 4);
        }

        uint32_t crc32_update(uint32_t crc, const unsigned char *buf, size_t len)
        {
            static const auto table = []()
            {
                std::array<uint32_t, 256> t{};
                for (uint32_t i = 0; i < 256; ++i)
                {
                    uint32_t c = i;
                    for (int j = 0; j < 8; ++j)
                        c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                    t[i] = c;
                }
                return t;
            }();
            for (size_t i = 0; i < len; ++i)
                crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
            return crc;
        }

        std::string make_npy_header(const cv::Mat &data)
        {
            CV_Assert(data.isContinuous() &&
                      (data.type() == CV_32FC1 || data.type() == CV_32FC3 || data.type() == CV_32FC4));

            std::string shape_str;
            if (data.channels() == 1)
                shape_str = "(" + std::to_string(data.rows) + ", " + std::to_string(data.cols) + "), }";
            else
                shape_str = "(" + std::to_string(data.rows) + ", " + std::to_string(data.cols) + ", " + std::to_string(data.channels()) + "), }";

            std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str;
            while ((10 + header.size() + 1) % 64 != 0)
                header += ' ';
            header += '\n';
            return header;
        }

        void save_npy(const std::string &path, const cv::Mat &data)
        {
            CV_Assert(!data.empty());
            CV_Assert(data.channels() == 1 || data.channels() == 3 || data.channels() == 4);

            cv::Mat prepared = data;
            if (prepared.depth() != CV_32F)
                prepared.convertTo(prepared, CV_32F);
            if (!prepared.isContinuous())
                prepared = prepared.clone();

            const std::string header = make_npy_header(prepared);
            std::ofstream f(path, std::ios::binary);
            const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                             static_cast<unsigned char>(header.size() & 0xff),
                                             static_cast<unsigned char>(header.size() >> 8)};
            f.write(reinterpret_cast<const char *>(magic), 10);
            f.write(header.data(), static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char *>(prepared.data),
                    static_cast<std::streamsize>(prepared.total() * prepared.elemSize()));
            if (!f)
                throw std::runtime_error("cannot write " + path);
        }


        void save_npz(const std::string &path, const std::string &entry_name, const cv::Mat &data)
        {
            CV_Assert(!data.empty());
            CV_Assert(data.channels() == 1 || data.channels() == 3 || data.channels() == 4);

            cv::Mat prepared = data;
            if (prepared.depth() != CV_32F)
                prepared.convertTo(prepared, CV_32F);
            if (!prepared.isContinuous())
                prepared = prepared.clone();

            const std::string filename_in_zip = (entry_name.size() >= 4 && entry_name.rfind(".npy") == entry_name.size() - 4)
                                                    ? entry_name
                                                    : entry_name + ".npy";
            const std::string header = make_npy_header(prepared);

            const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                             static_cast<unsigned char>(header.size() & 0xff),
                                             static_cast<unsigned char>(header.size() >> 8)};

            const size_t raw_data_size = prepared.total() * prepared.elemSize();
            const size_t npy_size = 10 + header.size() + raw_data_size;

            if (npy_size >= 0xFFFFFFFFULL)
            {
                std::fprintf(stderr,
                             "[warning] '%s': the array (%.2f GB) exceeds the 4 GB limit of standard ZIP; "
                             "no .npz written (use the .npy instead).\n",
                             path.c_str(), npy_size / 1073741824.0);
                return;
            }

            uint32_t crc = 0xFFFFFFFF;
            crc = crc32_update(crc, magic, 10);
            crc = crc32_update(crc, reinterpret_cast<const unsigned char *>(header.data()), header.size());
            crc = crc32_update(crc, reinterpret_cast<const unsigned char *>(prepared.data), raw_data_size);
            crc = ~crc;

            std::ofstream f(path, std::ios::binary);
            if (!f)
                throw std::runtime_error("cannot open for writing: " + path);

            const uint32_t u_npy_size = static_cast<uint32_t>(npy_size);
            const uint16_t fname_len = static_cast<uint16_t>(filename_in_zip.size());

            write_u32(f, 0x04034b50);
            write_u16(f, 20);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u32(f, crc);
            write_u32(f, u_npy_size);
            write_u32(f, u_npy_size);
            write_u16(f, fname_len);
            write_u16(f, 0);
            f.write(filename_in_zip.data(), fname_len);

            f.write(reinterpret_cast<const char *>(magic), 10);
            f.write(header.data(), static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char *>(prepared.data), static_cast<std::streamsize>(raw_data_size));

            const uint32_t cd_offset = 30 + fname_len + u_npy_size;

            write_u32(f, 0x02014b50);
            write_u16(f, 20);
            write_u16(f, 20);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u32(f, crc);
            write_u32(f, u_npy_size);
            write_u32(f, u_npy_size);
            write_u16(f, fname_len);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u32(f, 0);
            write_u32(f, 0);
            f.write(filename_in_zip.data(), fname_len);

            write_u32(f, 0x06054b50);
            write_u16(f, 0);
            write_u16(f, 0);
            write_u16(f, 1);
            write_u16(f, 1);
            write_u32(f, 46 + fname_len);
            write_u32(f, cd_offset);
            write_u16(f, 0);

            if (!f)
                throw std::runtime_error("cannot write " + path);
        }


        double tick_step(double span, int target)
        {
            const double raw = std::max(span, 1e-9) / target;
            const double power = std::pow(10.0, std::floor(std::log10(raw)));
            for (double m : {1.0, 2.0, 5.0, 10.0})
                if (raw <= m * power)
                    return m * power;
            return 10.0 * power;
        }

        void save_figure(const std::string &path, const cv::Mat &gray, const std::string &title, int z_start, int z_end,
                         const std::string &x_label, double x_max)
        {
            const int W = 2400, H = 1200, left = 190, right = 50, top = 90, bottom = 120;
            const cv::Scalar black(0, 0, 0);
            cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
            const cv::Rect plot(left, top, W - left - right, H - top - bottom);
            cv::Mat scaled, colour;
            const bool shrink = gray.cols >= plot.width && gray.rows >= plot.height;
            cv::resize(gray, scaled, plot.size(), 0, 0, shrink ? cv::INTER_AREA : cv::INTER_LINEAR);
            cv::cvtColor(scaled, colour, cv::COLOR_GRAY2BGR);
            colour.copyTo(canvas(plot));
            cv::rectangle(canvas, plot, black, 2);
            const int font = cv::FONT_HERSHEY_SIMPLEX;
            auto centred = [&](const std::string &text, int cx, int baseline, double scale, int thickness)
            {
                int base = 0;
                const cv::Size size = cv::getTextSize(text, font, scale, thickness, &base);
                cv::putText(canvas, text, cv::Point(cx - size.width / 2, baseline), font, scale, black, thickness, cv::LINE_AA);
            };
            centred(title, W / 2, 55, 1.0, 2);
            const double x_step = tick_step(x_max, 8);
            for (double v = 0; v <= x_max + 1e-9; v += x_step)
            {
                const int x = plot.x + static_cast<int>(std::lround(v / x_max * plot.width));
                cv::line(canvas, cv::Point(x, plot.br().y), cv::Point(x, plot.br().y + 10), black, 2);
                centred(std::to_string(static_cast<long long>(v)), x, plot.br().y + 42, 0.8, 1);
            }
            centred(x_label, plot.x + plot.width / 2, H - 25, 0.9, 2);
            const double z_lo = z_start - 0.5, z_hi = z_end + 0.5;
            const double z_step = std::max(1.0, tick_step(z_hi - z_lo, 8));
            for (double v = std::ceil(z_start / z_step) * z_step; v <= z_end; v += z_step)
            {
                const int y = plot.y + static_cast<int>(std::lround((v - z_lo) / (z_hi - z_lo) * plot.height));
                cv::line(canvas, cv::Point(plot.x - 10, y), cv::Point(plot.x, y), black, 2);
                const std::string text = std::to_string(static_cast<long long>(v));
                int base = 0;
                const cv::Size size = cv::getTextSize(text, font, 0.8, 1, &base);
                cv::putText(canvas, text, cv::Point(plot.x - 18 - size.width, y + size.height / 2), font, 0.8, black, 1,
                            cv::LINE_AA);
            }
            {
                const std::string text = "Slice Z (native voxels)";
                int base = 0;
                const cv::Size size = cv::getTextSize(text, font, 0.9, 2, &base);
                cv::Mat label(size.height + base + 8, size.width + 8, CV_8UC3, cv::Scalar(255, 255, 255));
                cv::putText(label, text, cv::Point(4, size.height + 4), font, 0.9, black, 2, cv::LINE_AA);
                cv::rotate(label, label, cv::ROTATE_90_COUNTERCLOCKWISE);
                const cv::Rect where(12, plot.y + (plot.height - label.rows) / 2, label.cols, label.rows);
                if ((where & cv::Rect(0, 0, W, H)) == where)
                    label.copyTo(canvas(where));
            }
            if (!cv::imwrite(path, canvas))
                throw std::runtime_error("cannot write " + path);
        }
    } // namespace

    std::string save_coverage_map(const cv::Mat &map, const std::string &output_png)
    {
        CV_Assert(map.type() == CV_8U);
        const std::filesystem::path out_path(output_png);
        if (!out_path.parent_path().empty())
            std::filesystem::create_directories(out_path.parent_path());
        const std::string path = (out_path.parent_path() / (out_path.stem().string() + "_coverage.png")).string();
        if (!cv::imwrite(path, map))
            throw std::runtime_error("cannot write " + path);
        return path;
    }

    void save_unrolled(const UnrolledImage &im, const std::string &output_png)
    {
        cv::Mat image = im.values.clone();
        const cv::Mat &coords = im.coords;
        std::vector<double> seen;
        for (int r = 0; r < image.rows; ++r)
        {
            const double *row = image.ptr<double>(r);
            for (int c = 0; c < image.cols; ++c)
                if (row[c] > 0)
                    seen.push_back(row[c]);
        }
        if (!seen.empty())
        {
            const double p3 = percentile(seen, 3.0), p97 = percentile(seen, 97.0);
            if (p97 > p3)
                for (int r = 0; r < image.rows; ++r)
                {
                    double *row = image.ptr<double>(r);
                    for (int c = 0; c < image.cols; ++c)
                        row[c] = row[c] > 0 ? std::min(std::max((row[c] - p3) / (p97 - p3), 0.0), 1.0) : 0.0;
                }
        }
        unsigned char lut[256];
        for (int i = 0; i < 256; ++i)
            lut[i] = static_cast<unsigned char>((i == 255 ? 1.0 : i * (1.0 / 255.0)) * 255.0);
        cv::Mat gray(image.size(), CV_8U);
        for (int r = 0; r < image.rows; ++r)
        {
            const double *row = image.ptr<double>(r);
            unsigned char *out = gray.ptr<unsigned char>(r);
            for (int c = 0; c < image.cols; ++c)
                out[c] = lut[std::min(std::max(static_cast<int>(row[c] * 256.0), 0), 255)];
        }

        const std::filesystem::path out_path(output_png);
        if (!out_path.parent_path().empty())
            std::filesystem::create_directories(out_path.parent_path());
        const std::string full_png =
            (out_path.parent_path() / (out_path.stem().string() + "_full.png")).string();
        if (!cv::imwrite(full_png, gray))
            throw std::runtime_error("cannot write " + full_png);

        const std::string coords_npy = (out_path.parent_path() / (out_path.stem().string() + "_coords.npy")).string();
        try
        {
            save_npy(coords_npy, coords);
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[error] Failed to save '%s': %s\n", coords_npy.c_str(), e.what());
        }

        const std::string coords_npz = (out_path.parent_path() / (out_path.stem().string() + "_coords.npz")).string();
        try
        {
            save_npz(coords_npz, "coords", coords);
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[error] Failed to save '%s': %s\n", coords_npz.c_str(), e.what());
        }

        if (!im.filled.empty() && !coords.empty())
        {
            const std::string meta_npy =
                (out_path.parent_path() / (out_path.stem().string() + "_coords_meta.npy")).string();
            try
            {
                cv::Mat meta(coords.size(), CV_32FC4, cv::Scalar::all(std::numeric_limits<float>::quiet_NaN()));
                for (int r = 0; r < meta.rows; ++r)
                {
                    const cv::Vec3f *c = coords.ptr<cv::Vec3f>(r);
                    const unsigned char *f = im.filled.ptr<unsigned char>(r);
                    cv::Vec4f *out = meta.ptr<cv::Vec4f>(r);
                    for (int col = 0; col < meta.cols; ++col)
                    {
                        if (!std::isfinite(c[col][0]))
                            continue;
                        out[col] = cv::Vec4f(c[col][0], c[col][1], c[col][2],
                                             f[col] ? 1.0f : 0.0f);
                    }
                }
                save_npy(meta_npy, meta);
                std::printf("--> coords_meta (4 channels x,y,z,filled): '%s'\n", meta_npy.c_str());
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "[error] Failed to save '%s': %s\n", meta_npy.c_str(), e.what());
            }
        }

        const std::string values_npy = (out_path.parent_path() / (out_path.stem().string() + "_values.npy")).string();
        const std::string values_npz = (out_path.parent_path() / (out_path.stem().string() + "_values.npz")).string();
        try
        {
            cv::Mat values_f32;
            im.values.convertTo(values_f32, CV_32FC1);

            try
            {
                save_npy(values_npy, values_f32);
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "[error] Failed to save '%s': %s\n", values_npy.c_str(), e.what());
            }

            try
            {
                save_npz(values_npz, "values", values_f32);
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "[error] Failed to save '%s': %s\n", values_npz.c_str(), e.what());
            }
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[error] Failed to convert/save values ('%s' / '%s'): %s\n",
                         values_npy.c_str(), values_npz.c_str(), e.what());
        }

        try
        {
            char range[64];
            const int z_start = im.z_start, z_end = im.z_end;
            std::snprintf(range, sizeof(range), " | Z = [%d -> %d]", z_start, z_end);
            save_figure(output_png, gray, im.title + range, z_start, z_end, im.x_label, im.x_max);
            std::printf("--> Image saved to: '%s' (full: '%s', coords.npy: '%s', coords.npz: '%s', values.npy: '%s', values.npz: '%s')\n",
                        output_png.c_str(), full_png.c_str(),
                        coords_npy.c_str(), coords_npz.c_str(),
                        values_npy.c_str(), values_npz.c_str());
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[error] Failed to save figure '%s': %s\n", output_png.c_str(), e.what());
        }

    }
} // namespace pw