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


struct TransposeDimensions {
    uint batch_size;
    uint input_size;
};

kernel void transpose(const device bc::Buffer* input [[ buffer(0) ]],
                      device bc::Buffer* output [[ buffer(1) ]],
                      constant TransposeDimensions& dims [[ buffer(2) ]],
                      uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;

    if (sizeElement >= dims.input_size || batchElement >= dims.batch_size)
        return;

    at(output, batchElement + sizeElement * dims.batch_size) = at(input, sizeElement + batchElement * dims.input_size);
}
