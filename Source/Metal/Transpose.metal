// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;


struct TransposeDimensions {
    uint batch_size;
    uint input_size;
};

kernel void transpose_forward(const device float* input [[ buffer(0) ]],
                                   device float* output [[ buffer(1) ]],
                                   constant TransposeDimensions& dims [[ buffer(2) ]],
                                   uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;

    if (sizeElement >= dims.input_size || batchElement >= dims.batch_size)
        return;

    output[batchElement + sizeElement * dims.batch_size] = input[sizeElement + batchElement * dims.input_size];
}
