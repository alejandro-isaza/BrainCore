// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>


inline float safe_tanh(const float x) {
    const auto P0 = -8.237728127e-1f;
    const auto P1 = -3.831010665e-3f;
    const auto Q0 = 2.471319654f;
    const auto xLarge = 8.66433975699931636772f;
    const auto xMedium = 5.4930614433405484570e-1f;
    const auto xSmall = 4.22863966691620432990e-4f;

    const auto sign = x > 0 ? 1 : -1;
    const auto absX = sign * x;

    if (absX >= xLarge) {
        return sign;
    } else if (xMedium <= x) {
        const auto temp = 0.5 - 1 / (1 + metal::exp(2 * absX));
        return sign * (temp + temp);
    } else if (xSmall <= x) {
        const auto g = absX * absX;
        const auto R = g * (P1 * g + P0) / (g + Q0);
        return x + x * R;
    } else {
        return sign * absX;
    }
}

inline float sigmoid(const float x) {
    return 1.0 / (1.0 + metal::exp(-x));
}

