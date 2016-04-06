// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>

#include "Utilities.h"

using namespace metal;

struct LSTMParameters {
    ushort batch_size;
    ushort unit_count;
    ushort input_size;
    float clip_to;
};

kernel void lstm_forward(const device float* input [[ buffer(0) ]],
                         const device float* weights [[ buffer(1) ]],
                         const device float* biases [[ buffer(2) ]],
                         device float* output [[ buffer(3) ]],
                         device float* state [[ buffer(4) ]],
                         constant LSTMParameters& params [[ buffer(5) ]],
                         uint2 id [[ thread_position_in_grid ]])
{
    const auto unit = id.x;
    const auto batchElement = id.y;

    if (unit >= params.unit_count || batchElement >= params.batch_size)
        return;

    const auto inputGateIndex  = 0 * params.unit_count + unit;
    const auto newInputIndex   = 1 * params.unit_count + unit;
    const auto forgetGateIndex = 2 * params.unit_count + unit;
    const auto outputGateIndex = 3 * params.unit_count + unit;

    auto inputGate  = biases[inputGateIndex];
    auto newInput   = biases[newInputIndex];
    auto forgetGate = biases[forgetGateIndex];
    auto outputGate = biases[outputGateIndex];

    for (uint i = 0; i < params.input_size; i += 1) {
        inputGate  += weights[inputGateIndex  + i * 4 * params.unit_count] * input[batchElement + i * params.batch_size];
        newInput   += weights[newInputIndex   + i * 4 * params.unit_count] * input[batchElement + i * params.batch_size];
        forgetGate += weights[forgetGateIndex + i * 4 * params.unit_count] * input[batchElement + i * params.batch_size];
        outputGate += weights[outputGateIndex + i * 4 * params.unit_count] * input[batchElement + i * params.batch_size];
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        inputGate  += weights[inputGateIndex  + j * 4 * params.unit_count] * state[params.unit_count * (1 + batchElement * 2) + i];
        newInput   += weights[newInputIndex   + j * 4 * params.unit_count] * state[params.unit_count * (1 + batchElement * 2) + i];
        forgetGate += weights[forgetGateIndex + j * 4 * params.unit_count] * state[params.unit_count * (1 + batchElement * 2) + i];
        outputGate += weights[outputGateIndex + j * 4 * params.unit_count] * state[params.unit_count * (1 + batchElement * 2) + i];
    }

    const auto previousActivation = state[unit + batchElement * 2 * params.unit_count];
    auto activation = bc::sigmoid(forgetGate + 1) * previousActivation + bc::sigmoid(inputGate) * bc::tanh(newInput);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = bc::sigmoid(outputGate) * bc::tanh(activation);

    output[batchElement + unit * params.batch_size] = out;
    state[unit + batchElement * 2 * params.unit_count] = activation;
    state[unit + params.unit_count * (1 + batchElement * 2)] = out;
}
