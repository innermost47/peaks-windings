// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "thinning.hpp"

#include "parallel.hpp"

#include <array>
#include <atomic>
#include <bitset>
#include <vector>

namespace pw
{
    int crossing_number(int code)
    {
        int runs = 0;
        for (int k = 0; k < 8; ++k)
            runs += !(code >> k & 1) && (code >> ((k + 1) % 8) & 1);
        return runs;
    }

    namespace
    {
        // Whether a pixel of neighbourhood `code` is removed by the first (pass 0) or second (pass 1) sub-iteration.
        std::array<std::array<unsigned char, 256>, 2> removal_tables()
        {
            std::array<std::array<unsigned char, 256>, 2> table{};
            for (int code = 0; code < 256; ++code)
            {
                auto set = [code](int k)
                { return code >> k & 1; }; // 0 N, 2 E, 4 S, 6 W
                const int count = static_cast<int>(std::bitset<8>(code).count());
                const bool simple = count >= 2 && count <= 6 && crossing_number(code) == 1;
                table[0][code] = simple && !(set(0) && set(2) && set(4)) && !(set(2) && set(4) && set(6));
                table[1][code] = simple && !(set(0) && set(2) && set(6)) && !(set(0) && set(4) && set(6));
            }
            return table;
        }
    } // namespace

    cv::Mat thin_lines(const cv::Mat &binary, int threads)
    {
        CV_Assert(binary.type() == CV_8U);
        static const auto table = removal_tables();
        cv::Mat img = (binary != 0) / 255;
        if (img.rows < 3 || img.cols < 3)
            return cv::Mat::zeros(binary.size(), CV_8U);
        img.row(0).setTo(0);
        img.row(img.rows - 1).setTo(0);
        img.col(0).setTo(0);
        img.col(img.cols - 1).setTo(0);
        const size_t step = img.step;
        const int inner_rows = img.rows - 2;
        std::vector<std::vector<int>> removed(inner_rows);
        for (bool changed = true; changed;)
        {
            changed = false;
            for (int pass = 0; pass < 2; ++pass)
            {
                // marks first, from the image as it was before this sub-iteration; then the removals
                std::atomic<bool> any{false};
                parallel_for(inner_rows, threads, [&](int i, int)
                             {
                                 std::vector<int> &cols = removed[i];
                                 cols.clear();
                                 const unsigned char *row = img.ptr<unsigned char>(i + 1);
                                 for (int c = 1; c + 1 < img.cols; ++c)
                                     if (row[c] && table[pass][neighbour_code(row + c, step)])
                                         cols.push_back(c);
                                 if (!cols.empty())
                                     any = true; });
                if (!any)
                    continue;
                changed = true;
                for (int i = 0; i < inner_rows; ++i)
                {
                    unsigned char *row = img.ptr<unsigned char>(i + 1);
                    for (int c : removed[i])
                        row[c] = 0;
                }
            }
        }
        return img;
    }
} // namespace pw
