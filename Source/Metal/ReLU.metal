// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;

kernel void linear_rectify_forward(const device float* input [[ buffer(0) ]],
                                   device float* output [[ buffer(1) ]],
                                   uint id [[ thread_position_in_grid ]])
{
    output[id] = fmax(0.0, input[id]);
}

kernel void linear_rectify_backward(const device float* outputDiff [[ buffer(0) ]],
                                    const device float* input [[ buffer(1) ]],
                                    device float* inputDiff [[ buffer(2) ]],
                                    uint id [[ thread_position_in_grid ]])
{
    if (input[id] > 0) {
        inputDiff[id] = outputDiff[id];
    } else {
        inputDiff[id] = 0.0;
    }
}
