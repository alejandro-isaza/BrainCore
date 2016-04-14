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
                         const device float* old_state [[ buffer(4) ]],
                         device float* new_state [[ buffer(5) ]],
                         constant LSTMParameters& params [[ buffer(6) ]],
                         uint2 id [[ thread_position_in_grid ]])
{
    const auto unit = id.x;
    const auto batch_element = id.y;

    if (unit >= params.unit_count || batch_element >= params.batch_size)
        return;

    const auto input_gate_index  = 0 * params.unit_count + unit;
    const auto new_input_index   = 1 * params.unit_count + unit;
    const auto forget_gate_index = 2 * params.unit_count + unit;
    const auto output_gate_index = 3 * params.unit_count + unit;

    auto input_gate  = biases[input_gate_index];
    auto new_input   = biases[new_input_index];
    auto forget_gate = biases[forget_gate_index];
    auto output_gate = biases[output_gate_index];

    for (uint i = 0; i < params.input_size; i += 1) {
        const auto input_value = input[batch_element + i * params.batch_size];
        input_gate  += weights[input_gate_index  + i * 4 * params.unit_count] * input_value;
        new_input   += weights[new_input_index   + i * 4 * params.unit_count] * input_value;
        forget_gate += weights[forget_gate_index + i * 4 * params.unit_count] * input_value;
        output_gate += weights[output_gate_index + i * 4 * params.unit_count] * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = old_state[params.unit_count * (1 + batch_element * 2) + i];
        input_gate  += weights[input_gate_index  + j * 4 * params.unit_count] * old_out;
        new_input   += weights[new_input_index   + j * 4 * params.unit_count] * old_out;
        forget_gate += weights[forget_gate_index + j * 4 * params.unit_count] * old_out;
        output_gate += weights[output_gate_index + j * 4 * params.unit_count] * old_out;
    }

    const auto old_activation = old_state[unit + batch_element * 2 * params.unit_count];
    auto activation = bc::sigmoid(forget_gate + 1) * old_activation + bc::sigmoid(input_gate) * bc::tanh(new_input);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = bc::sigmoid(output_gate) * bc::tanh(activation);

    output[batch_element + unit * params.batch_size] = out;
    new_state[unit + batch_element * 2 * params.unit_count] = activation;
    new_state[unit + params.unit_count * (1 + batch_element * 2)] = out;
}
