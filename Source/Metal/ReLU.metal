// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;


struct ReluDimensions {
    uint batch_size;
    uint size;
};

kernel void linear_rectify_forward(const device float* input [[ buffer(0) ]],
                                   device float* output [[ buffer(1) ]],
                                   constant ReluDimensions& dims [[ buffer(2) ]],
                                   uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;
    
    if (sizeElement >= dims.size || batchElement >= dims.batch_size)
        return;
    
    output[sizeElement + batchElement * dims.size] = fmax(0.0, input[sizeElement + batchElement * dims.size]);
}

kernel void linear_rectify_backward(const device float* outputDiff [[ buffer(0) ]],
                                    const device float* input [[ buffer(1) ]],
                                    device float* inputDiff [[ buffer(2) ]],
                                    constant ReluDimensions& dims [[ buffer(3) ]],
                                    uint2 id [[ thread_position_in_grid ]])
{
    const auto sizeElement = id.x;
    const auto batchElement = id.y;
    
    if (sizeElement >= dims.size || batchElement >= dims.batch_size)
        return;
    
    auto index = sizeElement + batchElement * dims.size;
    if (input[index] > 0) {
        inputDiff[index] = outputDiff[index];
    } else {
        inputDiff[index] = 0.0;
    }
}
