// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pw
{
    // A volume one tool read: where it is stored and its shape (z, y, x).
    struct VolumeRecord
    {
        std::string store;          // local path or URL, empty when unknown
        std::vector<int64_t> shape; // empty when unknown

        bool known() const { return !store.empty(); }
    };

    // The volumes of one scroll, kept next to its JSON results (<results dir>/scroll_settings.json) so every tool finds
    // them without being told again: `surface` is the surface prediction peaks_windings read (the voxels of every
    // result file), `ct` the CT volume unroll_layer and overlay_sections sample.
    class ScrollSettings
    {
    public:
        // The settings of the results in `results_dir` (all unknown when there is no file yet).
        static ScrollSettings load(const std::string &results_dir);
        static std::string file_in(const std::string &results_dir);

        // Writes the file (creating the folder), replacing the previous one in one step.
        void save() const;
        const std::string &file() const { return file_; }

        VolumeRecord surface, ct;

    private:
        std::string file_;
    };

    // The shape as text: "(z, y, x)".
    std::string shape_text(const std::vector<int64_t> &shape);

} // namespace pw
