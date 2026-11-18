// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once
#include <cmath>

namespace pw
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * PI;

    // Python's round(): half to even (the default floating-point rounding mode).
    inline int round_half_even(double v) { return static_cast<int>(std::nearbyint(v)); }

    // An angle brought into (-pi, pi].
    inline double wrap_angle(double a)
    {
        a = std::fmod(a + PI, TWO_PI);
        if (a <= 0)
            a += TWO_PI;
        return a - PI;
    }

    // The angle of (x, y) around (cx, cy), in [0, 2 pi).
    inline double angle_around(double x, double y, double cx, double cy)
    {
        const double a = std::atan2(y - cy, x - cx);
        return a < 0 ? a + TWO_PI : a;
    }
} // namespace pw
