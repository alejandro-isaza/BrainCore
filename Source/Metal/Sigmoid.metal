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


struct SigmoidDimensions {
    uint batch_size;
    uint size;
};

kernel void sigmoid_forward(const device float* input [[ buffer(0) ]],
                                   device float* output [[ buffer(1) ]],
                                   constant SigmoidDimensions& dims [[ buffer(2) ]],
                                   uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;

    if (sizeElement >= dims.size || batchElement >= dims.batch_size)
        return;

    const auto index = sizeElement + batchElement * dims.size;
    output[index] = bc::sigmoid(input[index]);
}

kernel void sigmoid_backward(const device float* outputDiff [[ buffer(0) ]],
                                    const device float* input [[ buffer(1) ]],
                                    device float* inputDiff [[ buffer(2) ]],
                                    constant SigmoidDimensions& dims [[ buffer(3) ]],
                                    uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;

    if (sizeElement >= dims.size || batchElement >= dims.batch_size)
        return;

    const auto index = sizeElement + batchElement * dims.size;
    inputDiff[index] = outputDiff[index] * bc::sigmoid(input[index]) * (1 - bc::sigmoid(input[index]));
}
