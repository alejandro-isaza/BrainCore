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
                            uint id [[ thread_position_in_grid ]])
{
    if (id >= dims.size * dims.batch_size)
        return;

    output[id] = bc::sigmoid(input[id]);
}

kernel void sigmoid_backward(const device float* outputDiff [[ buffer(0) ]],
                             const device float* input [[ buffer(1) ]],
                             device float* inputDiff [[ buffer(2) ]],
                             constant SigmoidDimensions& dims [[ buffer(3) ]],
                             uint id [[ thread_position_in_grid ]])
{
    if (id >= dims.size * dims.batch_size)
        return;

    inputDiff[id] = outputDiff[id] * bc::sigmoid(input[id]) * (1 - bc::sigmoid(input[id]));
}
