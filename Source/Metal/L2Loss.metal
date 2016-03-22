// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;


struct L2LossDimensions {
    ushort input_size;
    ushort batch_size;
};

kernel void l2_loss_forward(const device float* input [[ buffer(0) ]],
                            device float* output [[ buffer(1) ]],
                            constant L2LossDimensions& dims [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]])
{
    if (id >= 1) {
        return;
    }
    
    output[0] = 0;
    for (auto i = 0; i < dims.input_size / 2; ++i) {
        auto diff = input[i] - input[(dims.input_size / 2) + i];
        output[0] += diff * diff / 2;
    }
}

kernel void l2_loss_backward(const device float* input [[ buffer(0) ]],
                             device float* inputDiff [[ buffer(1) ]],
                             constant L2LossDimensions& dims [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]])
{
    if (id >= dims.input_size / 2) {
        return;
    }

    auto alpha = 1 / dims.batch_size;
    auto diff = input[id] - input[(dims.input_size / 2) + id];
    inputDiff[id] = alpha * diff;
    inputDiff[(dims.input_size / 2) + id] = alpha * -diff;
}
