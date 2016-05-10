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


struct ReluDimensions {
    uint batch_size;
    uint input_size;
};

kernel void linear_rectify_forward(const device bc::Buffer* input [[ buffer(0) ]],
                                   device bc::Buffer* output [[ buffer(1) ]],
                                   constant ReluDimensions& dims [[ buffer(2) ]],
                                   uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(input, id))
        return;
    
    at(output, id) = fmax(0.0, at(input, id));
}

kernel void linear_rectify_backward(const device bc::Buffer* outputDiff [[ buffer(0) ]],
                                    const device bc::Buffer* input [[ buffer(1) ]],
                                    device bc::Buffer* inputDiff [[ buffer(2) ]],
                                    constant ReluDimensions& dims [[ buffer(3) ]],
                                    uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(input, id))
        return;

    if (at(input, id) > 0) {
        at(inputDiff, id) = at(outputDiff, id);
    } else {
        at(inputDiff, id) = 0.0;
    }
}
