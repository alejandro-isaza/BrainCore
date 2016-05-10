// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

#include "Utilities.h"

using namespace metal;


struct L2LossDimensions {
    uint batch_size;
    uint input_size;
};

kernel void l2_loss_forward(const device bc::Buffer* input [[ buffer(0) ]],
                            device bc::Buffer* output [[ buffer(1) ]],
                            constant L2LossDimensions& dims [[ buffer(2) ]],
                            uint batchElement [[ thread_position_in_grid ]])
{
    if (batchElement >= dims.batch_size) {
        return;
    }

    output[batchElement] = 0.0;
    for (auto inputElement = uint(0); inputElement < dims.input_size; inputElement += 1) {
        const auto dataIndex = batchElement + inputElement * dims.batch_size;
        const auto labelIndex = dataIndex + dims.batch_size * dims.input_size;

        const auto diff = input[dataIndex] - input[labelIndex];
        output[batchElement] += diff * diff / 2;
    }
}

kernel void l2_loss_backward(const device bc::Buffer* input [[ buffer(0) ]],
                             device bc::Buffer* deltas [[ buffer(1) ]],
                             constant L2LossDimensions& dims [[ buffer(2) ]],
                             uint2 id [[ thread_position_in_grid ]])
{
    const auto inputElement = id.x;
    const auto batchElement = id.y;

    if (inputElement >= dims.input_size || batchElement >= dims.batch_size) {
        return;
    }

    const auto dataIndex = batchElement + inputElement * dims.batch_size;
    const auto labelIndex = dataIndex + dims.batch_size * dims.input_size;

    const auto alpha = 1.0 / dims.batch_size;
    const auto diff = input[dataIndex] - input[labelIndex];
    deltas[dataIndex] = alpha * diff;
    deltas[labelIndex] = alpha * -diff;
}
