// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>

#include "Utilities.h"

using namespace metal;
using namespace bc;

struct InnerProductDimensions {
    ushort batch_size;
    ushort input_size;
    ushort output_size;
};

kernel void inner_product_forward(const device Buffer* input [[ buffer(0) ]],
                                  device Buffer* output [[ buffer(1) ]],
                                  const device Buffer* weights [[ buffer(2) ]],
                                  const device Buffer* biases [[ buffer(3) ]],
                                  constant InnerProductDimensions& dims [[ buffer(4) ]],
                                  uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(output, id))
        return;

    const auto outputElement = id[2];
    const auto sequenceElement = id[1];
    const auto batchElement = id[0];

    at(output, id) = at(biases, outputElement);
    for (uint i = 0; i < dims.input_size; i += 1) {
        at(output, id) += at(weights, {i, outputElement}) * at(input, {i, sequenceElement, batchElement});
    }
}

kernel void inner_product_backward_params(const device Buffer* output_deltas [[ buffer(0) ]],
                                          const device Buffer* input [[ buffer(1) ]],
                                          device Buffer* weight_deltas [[ buffer(2) ]],
                                          device Buffer* bias_deltas [[ buffer(3) ]],
                                          constant InnerProductDimensions& dims [[ buffer(4) ]],
                                          uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(weight_deltas, id) || !isValid(weight_deltas, id))
        return;

    const auto outputElement = id[0];

    for (uint i = 0; i < dims.input_size; i += 1) {
        at(weight_deltas, {i, outputElement}) = 0.0;
    }
    at(bias_deltas, outputElement) = 0.0;
    for (uint sequenceElement = 0; sequenceElement < input->sequenceSize; sequenceElement += 1) {
        for (uint batchElement = 0; batchElement < input->batchSize; batchElement += 1) {
            for (uint inputElement = 0; inputElement < input->inputSize; inputElement += 1) {
                at(weight_deltas, {inputElement, outputElement}) += at(output_deltas, {outputElement, sequenceElement, batchElement}) * at(input, {inputElement, sequenceElement, batchElement});
            }
            at(bias_deltas, outputElement) += at(output_deltas, {outputElement, sequenceElement, batchElement});
        }
    }
}

kernel void inner_product_backward_input(const device Buffer* output_deltas [[ buffer(0) ]],
                                         device Buffer* input_deltas [[ buffer(1) ]],
                                         const device Buffer* weights [[ buffer(2) ]],
                                         constant InnerProductDimensions& dims [[ buffer(3) ]],
                                         uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(input_deltas, id))
        return;

    const auto inputElement = id[2];
    const auto batchElement = id[1];
    const auto sequenceElement = id[0];

    at(input_deltas, id) = 0.0;
    for (uint outputElement = 0; outputElement < dims.output_size; outputElement += 1) {
        at(input_deltas, id) += at(weights, {inputElement, outputElement}) * at(output_deltas, {outputElement, sequenceElement, batchElement});
    }
}
