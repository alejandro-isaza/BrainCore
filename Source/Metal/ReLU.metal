// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;

kernel void linear_rectify_forward(const device float* input,
                                   device float* output,
                                   uint id [[ thread_position_in_grid ]])
{
    output[id] = fmax(0.0, input[id]);
}

kernel void linear_rectify_backward(const device float* outputDiff,
                                    const device float* input,
                                    device float* inputDiff,
                                    uint id [[ thread_position_in_grid ]])
{
    if (input[id] > 0) {
        inputDiff[id] = outputDiff[id];
    } else {
        inputDiff[id] = 0.0;
    }
}
