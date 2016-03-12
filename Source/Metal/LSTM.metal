// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>

using namespace metal;

struct LSTMParameters {
    ushort batchSize;
    ushort unitCount;
    ushort inputSize;
    float clipTo;
};

inline float sigmoid(const float x) {
    return 1.0 / (1.0 + exp(-x));
}

kernel void lstm_forward(const device float* input [[ buffer(0) ]],
                         const device float* weights [[ buffer(1) ]],
                         const device float* biases [[ buffer(2) ]],
                         device float* output [[ buffer(3) ]],
                         device float* state [[ buffer(4) ]],
                         constant LSTMParameters& params [[ buffer(5) ]],
                         uint2 id [[ thread_position_in_grid ]])
{
    auto unit = id.x;
    if (unit >= params.unitCount || id.y >= params.batchSize)
        return;

    const auto inputGateIndex  = 0 * params.unitCount + unit;
    const auto newInputIndex   = 1 * params.unitCount + unit;
    const auto forgetGateIndex = 2 * params.unitCount + unit;
    const auto outputGateIndex = 3 * params.unitCount + unit;

    auto inputGate  = biases[inputGateIndex];
    auto newInput   = biases[newInputIndex];
    auto forgetGate = biases[forgetGateIndex];
    auto outputGate = biases[outputGateIndex];

    for (uint i = 0; i < params.inputSize; i += 1) {
        inputGate  += weights[inputGateIndex  + i * 4 * params.unitCount] * input[i + id.y * params.inputSize];
        newInput   += weights[newInputIndex   + i * 4 * params.unitCount] * input[i + id.y * params.inputSize];
        forgetGate += weights[forgetGateIndex + i * 4 * params.unitCount] * input[i + id.y * params.inputSize];
        outputGate += weights[outputGateIndex + i * 4 * params.unitCount] * input[i + id.y * params.inputSize];
    }
    for (uint i = 0; i < params.unitCount; i += 1) {
        const auto j = i + params.inputSize;
        inputGate  += weights[inputGateIndex  + j * 4 * params.unitCount] * state[params.unitCount * (id.y * 2 + 1) + i];
        newInput   += weights[newInputIndex   + j * 4 * params.unitCount] * state[params.unitCount * (id.y * 2 + 1) + i];
        forgetGate += weights[forgetGateIndex + j * 4 * params.unitCount] * state[params.unitCount * (id.y * 2 + 1) + i];
        outputGate += weights[outputGateIndex + j * 4 * params.unitCount] * state[params.unitCount * (id.y * 2 + 1) + i];
    }

    const auto previousActivation = state[unit];
    auto activation = sigmoid(forgetGate + 1) * previousActivation + sigmoid(inputGate) * tanh(newInput);
    if (params.clipTo > 0) {
        activation = clamp(activation, -params.clipTo, params.clipTo);
    }
    const auto out = sigmoid(outputGate) * tanh(activation);

    output[unit + id.y * params.unitCount] = out;
    state[unit + id.y * 2 * params.unitCount] = activation;
    state[unit + params.unitCount + id.y * 2 * params.unitCount] = out;
}
