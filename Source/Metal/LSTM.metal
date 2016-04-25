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
                         device float* activations [[ buffer(4) ]],
                         const device float* previous_state [[ buffer(5) ]],
                         device float* state [[ buffer(6) ]],
                         const device uint* time [[ buffer(7) ]],
                         constant LSTMParameters& params [[ buffer(8) ]],
                         uint2 id [[ thread_position_in_grid ]])
{
    const auto unit = id.x;
    const auto batch_element = id.y;

    if (unit >= params.unit_count || batch_element >= params.batch_size)
        return;

    const auto input_gate_index       = 0 * params.unit_count + unit;
    const auto input_activation_index = 1 * params.unit_count + unit;
    const auto forget_gate_index      = 2 * params.unit_count + unit;
    const auto output_gate_index      = 3 * params.unit_count + unit;

    auto input_gate       = biases[input_gate_index];
    auto input_activation = biases[input_activation_index];
    auto forget_gate      = biases[forget_gate_index];
    auto output_gate      = biases[output_gate_index];

    for (uint i = 0; i < params.input_size; i += 1) {
        const auto input_value = input[*time * params.batch_size * params.input_size + batch_element + i * params.batch_size];
        input_gate       += weights[input_gate_index       + i * 4 * params.unit_count] * input_value;
        input_activation += weights[input_activation_index + i * 4 * params.unit_count] * input_value;
        forget_gate      += weights[forget_gate_index      + i * 4 * params.unit_count] * input_value;
        output_gate      += weights[output_gate_index      + i * 4 * params.unit_count] * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = previous_state[params.unit_count * (1 + batch_element * 2) + i];
        input_gate       += weights[input_gate_index       + j * 4 * params.unit_count] * old_out;
        input_activation += weights[input_activation_index + j * 4 * params.unit_count] * old_out;
        forget_gate      += weights[forget_gate_index      + j * 4 * params.unit_count] * old_out;
        output_gate      += weights[output_gate_index      + j * 4 * params.unit_count] * old_out;
    }

    const auto old_activation = previous_state[unit + batch_element * 2 * params.unit_count];
    auto activation = bc::sigmoid(forget_gate) * old_activation + bc::sigmoid(input_gate) * bc::tanh(input_activation);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = bc::sigmoid(output_gate) * bc::tanh(activation);

    output[*time * params.batch_size * params.unit_count + batch_element + unit * params.batch_size] = out;
    state[unit + batch_element * 2 * params.unit_count] = activation;
    state[unit + params.unit_count * (1 + batch_element * 2)] = out;

    activations[input_gate_index       + batch_element * 4 * params.unit_count] = input_gate;
    activations[input_activation_index + batch_element * 4 * params.unit_count] = input_activation;
    activations[forget_gate_index      + batch_element * 4 * params.unit_count] = forget_gate;
    activations[output_gate_index      + batch_element * 4 * params.unit_count] = output_gate;
}

kernel void lstm_backward_activations(const device float* output_delta [[ buffer(0) ]],
                                      const device float* weights [[ buffer(1) ]],
                                      const device float* activations [[ buffer(2) ]],
                                      const device float* future_activations [[ buffer(3) ]],
                                      device float* activation_delta [[ buffer(4) ]],
                                      const device float* future_activation_delta [[ buffer(5) ]],
                                      const device float* state [[ buffer(6) ]],
                                      device float* state_delta [[ buffer(7) ]],
                                      const device float* old_state [[ buffer(8) ]],
                                      const device float* future_state_delta [[ buffer(9) ]],
                                      const device uint* time [[ buffer(10) ]],
                                      constant LSTMParameters& params [[ buffer(11) ]],
                                      uint2 id [[ thread_position_in_grid ]])
{
    const auto unit = id.x;
    const auto batch_element = id.y;

    if (unit >= params.unit_count || batch_element >= params.batch_size)
        return;

    const auto input_gate_index        = 0 * params.unit_count + unit;
    const auto input_activation_index  = 1 * params.unit_count + unit;
    const auto forget_gate_index       = 2 * params.unit_count + unit;
    const auto output_gate_index       = 3 * params.unit_count + unit;

    // (T + 1)'s activation deltas
    const auto future_input_gate_delta       = future_activation_delta[batch_element * 4 * params.unit_count + input_gate_index];
    const auto future_input_activation_delta = future_activation_delta[batch_element * 4 * params.unit_count + input_activation_index];
    const auto future_forget_gate_delta      = future_activation_delta[batch_element * 4 * params.unit_count + forget_gate_index];
    const auto future_output_gate_delta      = future_activation_delta[batch_element * 4 * params.unit_count + output_gate_index];

    // Unrolled output delta = output_delta + (unit_weights ⨉ future_activation_deltas)
    auto true_ouput_delta = output_delta[*time * params.batch_size * params.unit_count + batch_element + unit * params.batch_size];
    true_ouput_delta     += weights[input_gate_index       + (params.input_size + unit) * 4 * params.unit_count] * future_input_gate_delta;
    true_ouput_delta     += weights[input_activation_index + (params.input_size + unit) * 4 * params.unit_count] * future_input_activation_delta;
    true_ouput_delta     += weights[forget_gate_index      + (params.input_size + unit) * 4 * params.unit_count] * future_forget_gate_delta;
    true_ouput_delta     += weights[output_gate_index      + (params.input_size + unit) * 4 * params.unit_count] * future_output_gate_delta;

    // T's unsmoothed activations
    const auto unsmoothed_input_gate       = activations[batch_element * 4 * params.unit_count + input_gate_index];
    const auto unsmoothed_input_activation = activations[batch_element * 4 * params.unit_count + input_activation_index];
    const auto unsmoothed_forget_gate      = activations[batch_element * 4 * params.unit_count + forget_gate_index];
    const auto unsmoothed_output_gate      = activations[batch_element * 4 * params.unit_count + output_gate_index];

    // T's smoothed activations
    const auto smoothed_input_gate       = bc::sigmoid(unsmoothed_input_gate);
    const auto smoothed_input_activation = bc::tanh(unsmoothed_input_activation);
    const auto smoothed_forget_gate      = bc::sigmoid(unsmoothed_forget_gate);
    const auto smoothed_output_gate      = bc::sigmoid(unsmoothed_output_gate);

    // State delta
    /*
     d_state = true_output_delta ⊙ smooth_output_gate ⊙ (1 − tanh^2(state)) + d_future_state ⊙ smoothed_future_forget_gate
     */
    state_delta[unit + batch_element * 2 * params.unit_count] = true_ouput_delta * smoothed_output_gate * (1 - bc::tanh(state[unit + batch_element * 2 * params.unit_count]) * bc::tanh(state[unit + batch_element * 2 * params.unit_count])) + future_state_delta[unit + batch_element * 2 * params.unit_count] * bc::sigmoid(future_activations[batch_element * 4 * params.unit_count + forget_gate_index]);

    // Delta of unsmoothed functions
    /*
     d_unsmoothed_i = d_state       ⊙ smooth_input_activation
     d_unsmoothed_a = d_state       ⊙ smooth_input_gate
     d_unsmoothed_f = d_state       ⊙ old_state
     d_unsmoothed_o = d_true_output ⊙ tanh(state)
     */
    const auto unsmoothed_input_gate_delta       = state_delta[unit + batch_element * 2 * params.unit_count] * smoothed_input_activation;
    const auto unsmoothed_input_activation_delta = state_delta[unit + batch_element * 2 * params.unit_count] * smoothed_input_gate;
    const auto unsmoothed_forget_gate_delta      = state_delta[unit + batch_element * 2 * params.unit_count] * old_state[unit + batch_element * 2 * params.unit_count];
    const auto unsmoothed_output_gate_delta      = true_ouput_delta * bc::tanh(state[unit + batch_element * 2 * params.unit_count]);

    // Delta of smoothed functions
    /*
     d_smoothed_i = d_smoothed_i   ⊙ smoothed_input_gate  ⊙ (1 - smoothed_input_gate)
     d_smoothed_a = d_unsmoothed_a ⊙ (1 - (smoothed_input_activation)^2)
     d_smoothed_f = d_smoothed_f   ⊙ smoothed_forget_gate ⊙ (1 - smoothed_forget_gate)
     d_smoothed_o = d_smoothed_o   ⊙ smoothed_output_gate ⊙ (1 - smoothed_output_gate)
     */
    activation_delta[batch_element * 4 * params.unit_count + input_gate_index]       = unsmoothed_input_gate_delta       * smoothed_input_gate  * (1 - smoothed_input_gate);
    activation_delta[batch_element * 4 * params.unit_count + input_activation_index] = unsmoothed_input_activation_delta * (1 - smoothed_input_activation * smoothed_input_activation);
    activation_delta[batch_element * 4 * params.unit_count + forget_gate_index]      = unsmoothed_forget_gate_delta      * smoothed_forget_gate * (1 - smoothed_forget_gate);
    activation_delta[batch_element * 4 * params.unit_count + output_gate_index]      = unsmoothed_output_gate_delta      * smoothed_output_gate * (1 - smoothed_output_gate);

    // Delta of previous input
    state_delta[unit + params.unit_count + batch_element * 2 * params.unit_count]  = weights[input_gate_index + (params.input_size + unit) * 4 * params.unit_count] * activation_delta[batch_element * 4 * params.unit_count + input_gate_index];
    state_delta[unit + params.unit_count + batch_element * 2 * params.unit_count] += weights[input_activation_index + (params.input_size + unit) * 4 * params.unit_count] * activation_delta[batch_element * 4 * params.unit_count + input_activation_index];
    state_delta[unit + params.unit_count + batch_element * 2 * params.unit_count] += weights[forget_gate_index + (params.input_size + unit) * 4 * params.unit_count] * activation_delta[batch_element * 4 * params.unit_count + forget_gate_index];
    state_delta[unit + params.unit_count + batch_element * 2 * params.unit_count] += weights[output_gate_index + (params.input_size + unit) * 4 * params.unit_count] * activation_delta[batch_element * 4 * params.unit_count + output_gate_index];
}

kernel void lstm_backward_weights(const device float* input [[ buffer(0) ]],
                                  const device float* state [[ buffer(1) ]],
                                  device float* weights_delta [[ buffer(2) ]],
                                  device float* bias_delta [[ buffer(3) ]],
                                  const device float* activation_delta [[ buffer(4) ]],
                                  const device float* future_activation_delta [[ buffer(5) ]],
                                  const device uint* time [[ buffer(6) ]],
                                  constant LSTMParameters& params [[ buffer(7) ]],
                                  uint2 id [[ thread_position_in_grid ]])
{
    const auto unit = id.x;
    const auto batch_element = id.y;

    if (unit >= params.unit_count || batch_element >= params.batch_size)
        return;

    const auto input_gate_index       = 0 * params.unit_count + unit;
    const auto input_activation_index = 1 * params.unit_count + unit;
    const auto forget_gate_index      = 2 * params.unit_count + unit;
    const auto output_gate_index      = 3 * params.unit_count + unit;

    for (auto input_element = 0; input_element < params.input_size; input_element += 1) {
        const auto input_value = input[*time * params.batch_size * params.input_size + batch_element + input_element * params.batch_size];
        weights_delta[input_gate_index       + input_element * 4 * params.unit_count] += activation_delta[batch_element * 4 * params.unit_count + input_gate_index]       * input_value;
        weights_delta[input_activation_index + input_element * 4 * params.unit_count] += activation_delta[batch_element * 4 * params.unit_count + input_activation_index] * input_value;
        weights_delta[forget_gate_index      + input_element * 4 * params.unit_count] += activation_delta[batch_element * 4 * params.unit_count + forget_gate_index]      * input_value;
        weights_delta[output_gate_index      + input_element * 4 * params.unit_count] += activation_delta[batch_element * 4 * params.unit_count + output_gate_index]      * input_value;
    }
    for (auto unit = 0; unit < params.unit_count; unit += 1) {
        const auto j = unit + params.input_size;
        const auto old_out = state[params.unit_count * (1 + batch_element * 2) + unit];
        weights_delta[input_gate_index       + j * 4 * params.unit_count] += future_activation_delta[batch_element * 4 * params.unit_count + input_gate_index]       * old_out;
        weights_delta[input_activation_index + j * 4 * params.unit_count] += future_activation_delta[batch_element * 4 * params.unit_count + input_activation_index] * old_out;
        weights_delta[forget_gate_index      + j * 4 * params.unit_count] += future_activation_delta[batch_element * 4 * params.unit_count + forget_gate_index]      * old_out;
        weights_delta[output_gate_index      + j * 4 * params.unit_count] += future_activation_delta[batch_element * 4 * params.unit_count + output_gate_index]      * old_out;

    }

    bias_delta[input_gate_index]       = activation_delta[batch_element * 4 * params.unit_count + input_gate_index];
    bias_delta[input_activation_index] = activation_delta[batch_element * 4 * params.unit_count + input_activation_index];
    bias_delta[forget_gate_index]      = activation_delta[batch_element * 4 * params.unit_count + forget_gate_index];
    bias_delta[output_gate_index]      = activation_delta[batch_element * 4 * params.unit_count + output_gate_index];
}

kernel void lstm_backward_inputs(device float* input_delta [[ buffer(0) ]],
                                 const device float* weights [[ buffer(1) ]],
                                 const device float* activation_delta [[ buffer(2) ]],
                                 const device uint* time [[ buffer(3) ]],
                                 constant LSTMParameters& params [[ buffer(4) ]],
                                 uint2 id [[ thread_position_in_grid ]])
{
    const auto input_element = id.x;
    const auto batch_element = id.y;

    if (input_element >= params.input_size || batch_element >= params.batch_size)
        return;

    const auto element_index = batch_element + input_element * params.batch_size;

    for (auto unit = 0; unit < params.unit_count; unit += 1) {
        const auto input_gate_index       = 0 * params.unit_count + unit;
        const auto input_activation_index = 1 * params.unit_count + unit;
        const auto forget_gate_index      = 2 * params.unit_count + unit;
        const auto output_gate_index      = 3 * params.unit_count + unit;

        input_delta[*time * params.batch_size * params.input_size + element_index]  = weights[input_gate_index + input_element * 4 * params.unit_count]       * activation_delta[batch_element * 4 * params.unit_count + input_gate_index];
        input_delta[*time * params.batch_size * params.input_size + element_index] += weights[input_activation_index + input_element * 4 * params.unit_count] * activation_delta[batch_element * 4 * params.unit_count + input_activation_index];
        input_delta[*time * params.batch_size * params.input_size + element_index] += weights[forget_gate_index + input_element * 4 * params.unit_count]      * activation_delta[batch_element * 4 * params.unit_count + forget_gate_index];
        input_delta[*time * params.batch_size * params.input_size + element_index] += weights[output_gate_index + input_element * 4 * params.unit_count]      * activation_delta[batch_element * 4 * params.unit_count + output_gate_index];
    }
}
