// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "layers.hpp"
#include "turbo.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    using pw::Point;

    class TextFile
    {
    public:
        explicit TextFile(const std::string &path) : path_(path), file_(std::fopen(path.c_str(), "wb"))
        {
            if (!file_)
                throw std::runtime_error("cannot write " + path);
            std::setvbuf(file_, nullptr, _IOFBF, 1 << 22);
        }
        ~TextFile()
        {
            if (file_)
                std::fclose(file_);
        }
        std::FILE *get() { return file_; }
        void close()
        {
            const bool failed = std::ferror(file_) != 0;
            const bool close_failed = std::fclose(file_) != 0;
            file_ = nullptr;
            if (failed || close_failed)
                throw std::runtime_error("cannot write " + path_);
        }

    private:
        std::string path_;
        std::FILE *file_;
    };

    void make_parent(const std::string &path)
    {
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
    }

    void export_ply_pointcloud(const std::string &path, const std::vector<Point> &points)
    {
        if (points.empty())
            return;
        int min_w = points[0].winding, max_w = points[0].winding;
        for (const Point &p : points)
        {
            min_w = std::min(min_w, p.winding);
            max_w = std::max(max_w, p.winding);
        }
        const double span = std::max(1.0, static_cast<double>(max_w - min_w));
        make_parent(path);
        TextFile out(path);
        std::fprintf(out.get(),
                     "ply\nformat ascii 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
                     points.size());
        for (const Point &p : points)
        {
            const double norm = (p.winding - min_w) / span;
            const int index = std::min(static_cast<int>(norm * 256.0), 255);
            const unsigned char *c = pw::TURBO[index];
            std::fprintf(out.get(), "%.2f %.2f %.2f %d %d %d\n", p.x, p.y, static_cast<double>(p.z), c[0], c[1], c[2]);
        }
        out.close();
        std::printf("--> Saved 3D point cloud: '%s' (%zu vertices)\n", path.c_str(), points.size());
    }

    void export_obj_mesh(const std::string &path, const std::vector<Point> &points, size_t min_points_layer)
    {
        std::map<std::pair<int, int>, std::vector<size_t>> by_layer;
        for (size_t i = 0; i < points.size(); ++i)
            by_layer[{points[i].winding, points[i].track}].push_back(i);
        size_t viable = 0;
        for (const auto &layer : by_layer)
            viable += layer.second.size() >= min_points_layer;
        std::printf("--> Reconstructing 3D surface for %zu coherent layers (layer, summit)...\n", viable);

        std::vector<size_t> vertices;
        std::vector<std::array<size_t, 3>> triangles;
        for (const auto &[w, layer] : by_layer)
        {
            if (layer.size() < min_points_layer)
                continue;

            std::map<int, std::map<int, size_t>> slice_map;
            for (size_t i : layer)
                slice_map[points[i].z][points[i].ray] = i;

            std::map<std::pair<int, int>, size_t> vertex_id;
            auto vid = [&](size_t i)
            {
                const auto key = std::make_pair(points[i].z, points[i].ray);
                const auto it = vertex_id.find(key);
                if (it != vertex_id.end())
                    return it->second;
                vertices.push_back(i);
                vertex_id[key] = vertices.size();
                return vertices.size();
            };

            for (auto a = slice_map.begin(); a != slice_map.end(); ++a)
            {
                const auto b = std::next(a);
                if (b == slice_map.end())
                    break;
                std::vector<int> common;
                for (const auto &ray : a->second)
                    if (b->second.count(ray.first))
                        common.push_back(ray.first);
                for (size_t i = 0; i + 1 < common.size(); ++i)
                {
                    const int r1 = common[i], r2 = common[i + 1];
                    if (r2 - r1 > 3)
                        continue; // larger gaps are holes in the layer
                    const size_t a1 = vid(a->second.at(r1)), a2 = vid(a->second.at(r2));
                    const size_t b1 = vid(b->second.at(r1)), b2 = vid(b->second.at(r2));
                    triangles.push_back({a1, a2, b2});
                    triangles.push_back({a1, b2, b1});
                }
            }
        }

        make_parent(path);
        TextFile out(path);
        std::fprintf(out.get(), "# Sheet surfaces, one mesh per winding layer (peaks_windings)\n\n");
        for (size_t i : vertices)
            std::fprintf(out.get(), "v %.2f %.2f %.2f\n", points[i].x, points[i].y, static_cast<double>(points[i].z));
        std::fprintf(out.get(), "\n");
        for (const auto &t : triangles)
            std::fprintf(out.get(), "f %zu %zu %zu\n", t[0], t[1], t[2]);
        out.close();
        std::printf("--> Saved 3D surface mesh: '%s' (%zu vertices, %zu triangles)\n", path.c_str(), vertices.size(),
                    triangles.size());
    }

    void usage()
    {
        std::cout << "Convert scan results into a 3D OBJ mesh and a PLY point cloud.\n\n"
                     "  --scroll-name NAME  name of the scroll, used in the file names and folders (default scroll)\n"
                     "  --results-dir DIR   folder containing the JSON results (default: results/json/NAME)\n"
                     "  --min-points N      minimum number of points of a layer seen from a summit to mesh it (default: 60)\n"
                     "  --track N           keep only the rays of this summit (default: all)\n"
                     "  --z-min N           first slice kept (default: all)\n"
                     "  --z-max N           last slice kept (default: all)\n"
                     "  --threads N         threads reading the JSON files (default 0: all)\n";
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        std::string results_dir;
        std::string output_dir = "results/3D";
        std::string scroll_name = "scroll";
        long min_points = 60;
        int z_min = INT_MIN, z_max = INT_MAX, threads = 0;
        bool has_track = false;
        int track = 0;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                usage();
                return 0;
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
            if (arg == "--results-dir")
                results_dir = value;
            else if (arg == "--scroll-name")
                scroll_name = value;
            else if (arg == "--min-points")
                min_points = std::stol(value);
            else if (arg == "--z-min")
                z_min = std::stoi(value);
            else if (arg == "--z-max")
                z_max = std::stoi(value);
            else if (arg == "--threads")
                threads = std::stoi(value);
            else if (arg == "--track")
                track = std::stoi(value), has_track = true;
            else
                throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
        // Paths that depend on the scroll name are built once every option is known
        if (results_dir.empty())
            results_dir = "results/json/" + scroll_name;
        const std::string obj_out = output_dir + "/obj/scroll_" + scroll_name + ".obj";
        const std::string ply_out = output_dir + "/ply/" + scroll_name + "_pointcloud.ply";

        // 1. Load results, labels aligned along Z
        std::vector<Point> points = pw::load_points(results_dir, z_min, z_max, threads);
        pw::align_labels(points);
        if (has_track)
            points.erase(std::remove_if(points.begin(), points.end(), [&](const Point &p)
                                        { return p.track != track; }),
                         points.end());
        std::set<int> slices_z;
        for (const Point &p : points)
            slices_z.insert(p.z);
        std::printf("--> Loaded %zu 3D points across %zu Z slices.\n", points.size(), slices_z.size());

        // 2. PLY point cloud, 3. OBJ surface mesh
        export_ply_pointcloud(ply_out, points);
        export_obj_mesh(obj_out, points, static_cast<size_t>(std::max(min_points, 0L)));

        const std::string bar(65, '=');
        std::printf("\n%s\n                 3D EXPORT DONE\n%s\n", bar.c_str(), bar.c_str());
        std::printf(" 1. Colored Point Cloud (.ply) : %s\n", std::filesystem::absolute(ply_out).string().c_str());
        std::printf(" 2. 3D Surface Mesh     (.obj) : %s\n", std::filesystem::absolute(obj_out).string().c_str());
        std::printf("%s\nBoth files open in Blender, MeshLab or CloudCompare.\n\n", bar.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
