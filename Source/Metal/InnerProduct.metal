// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>

using namespace metal;

struct InnerProductDimensions {
    ushort batch_size;
    ushort input_size;
    ushort output_size;
};

kernel void inner_product_forward(const device float* input [[ buffer(0) ]],
                                  const device float* weights [[ buffer(1) ]],
                                  const device float* biases [[ buffer(2) ]],
                                  device float* output [[ buffer(3) ]],
                                  constant InnerProductDimensions& dims [[ buffer(4) ]],
                                  uint2 id [[ thread_position_in_grid ]])
{
    if (id.x >= dims.output_size || id.y >= dims.batch_size)
        return;
    
    output[id.x + id.y * dims.output_size] = biases[id.x];
    for (uint i = 0; i < dims.input_size; i += 1) {
        output[id.x + id.y * dims.output_size] += weights[id.x + i * dims.output_size] * input[i + id.y * dims.input_size];
    }
}

kernel void inner_product_backward_params(const device float* outputDiff [[ buffer(0) ]],
                                          const device float* input [[ buffer(1) ]],
                                          device float* weightDiff [[ buffer(2) ]],
                                          device float* biasDiff [[ buffer(3) ]],
                                          constant InnerProductDimensions& dims [[ buffer(4) ]],
                                          uint id [[ thread_position_in_grid ]])
{
    if (id >= dims.output_size)
        return;
    for (uint i = 0; i < dims.batch_size; i += 1) {
        for (uint j = 0; j < dims.input_size; j += 1) {
            weightDiff[id +  j * dims.output_size] += outputDiff[id + i * dims.output_size] * input[j + i * dims.input_size];
        }
        biasDiff[id] += outputDiff[id + i * dims.output_size];
    }
}

kernel void inner_product_backward_input(const device float* outputDiff [[ buffer(0) ]],
                                         const device float* weights [[ buffer(1) ]],
                                         device float* inputDiff [[ buffer(2) ]],
                                         constant InnerProductDimensions& dims [[ buffer(3) ]],
                                         uint2 id [[ thread_position_in_grid ]])
{
    if (id.x >= dims.input_size || id.y >= dims.batch_size)
        return;

    for (uint i = 0; i < dims.output_size; i += 1) {
        inputDiff[id.x + id.y * dims.input_size] = weights[i + id.x * dims.output_size] * outputDiff[i + id.y * dims.output_size];
    }
}
