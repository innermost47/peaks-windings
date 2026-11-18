// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "scroll_settings.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pw
{
    namespace
    {
        using json = nlohmann::json;
        namespace fs = std::filesystem;

        VolumeRecord record_from(const json &j, const char *key)
        {
            VolumeRecord r;
            if (!j.contains(key) || !j[key].is_object())
                return r;
            const json &v = j[key];
            r.store = v.value("store", std::string());
            if (v.contains("shape") && v["shape"].is_array())
                r.shape = v["shape"].get<std::vector<int64_t>>();
            return r;
        }

        json record_to(const VolumeRecord &r)
        {
            json v = {{"store", r.store}};
            if (!r.shape.empty())
                v["shape"] = r.shape;
            return v;
        }
    } // namespace

    std::string ScrollSettings::file_in(const std::string &results_dir)
    {
        return (fs::u8path(results_dir) / "scroll_settings.json").u8string();
    }

    ScrollSettings ScrollSettings::load(const std::string &results_dir)
    {
        ScrollSettings s;
        s.file_ = file_in(results_dir);
        std::ifstream f(fs::u8path(s.file_), std::ios::binary);
        if (!f)
            return s;
        json j;
        try
        {
            f >> j;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(s.file_ + " is not valid JSON: " + e.what());
        }
        s.surface = record_from(j, "surface");
        s.ct = record_from(j, "ct");
        return s;
    }

    void ScrollSettings::save() const
    {
        json j = json::object();
        if (surface.known())
            j["surface"] = record_to(surface);
        if (ct.known())
            j["ct"] = record_to(ct);
        const fs::path path = fs::u8path(file_);
        if (path.has_parent_path())
            fs::create_directories(path.parent_path());
        const fs::path temp = fs::u8path(file_ + ".tmp");
        {
            std::ofstream f(temp, std::ios::binary | std::ios::trunc);
            f << j.dump(2) << "\n";
            if (!f)
                throw std::runtime_error("cannot write " + temp.u8string());
        }
        fs::rename(temp, path);
    }

    std::string shape_text(const std::vector<int64_t> &shape)
    {
        std::ostringstream s;
        s << "(";
        for (size_t i = 0; i < shape.size(); ++i)
            s << (i ? ", " : "") << shape[i];
        s << ")";
        return s.str();
    }

} // namespace pw
