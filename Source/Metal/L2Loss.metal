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
                            uint batchElement [[ thread_position_in_grid ]])
{
    if (batchElement >= dims.batch_size) {
        return;
    }

    for (auto inputElement = 0; inputElement < dims.input_size / 2; ++inputElement) {
        const auto dataIndex = inputElement + batchElement * dims.input_size / 2;
        const auto labelIndex = dataIndex + dims.batch_size * dims.input_size / 2;

        const auto diff = input[dataIndex] - input[labelIndex];
        output[batchElement] += diff * diff / 2;
    }
}

kernel void l2_loss_backward(const device float* input [[ buffer(0) ]],
                             device float* inputDiff [[ buffer(1) ]],
                             constant L2LossDimensions& dims [[ buffer(2) ]],
                             uint2 id [[ thread_position_in_grid ]])
{
    const auto inputElement = id.x;
    const auto batchElement = id.y;

    if (inputElement >= dims.input_size / 2 || batchElement >= dims.batch_size) {
        return;
    }

    const auto dataIndex = inputElement + batchElement * dims.input_size / 2;
    const auto labelIndex = dataIndex + dims.batch_size * dims.input_size / 2;

    const auto alpha = 1.0 / dims.batch_size;
    const auto diff = input[dataIndex] - input[labelIndex];
    inputDiff[dataIndex] = alpha * diff;
    inputDiff[labelIndex] = alpha * -diff;
}
