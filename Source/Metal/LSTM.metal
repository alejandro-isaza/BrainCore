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

struct LSTMParameters {
    ushort batch_size;
    ushort unit_count;
    ushort input_size;
    float clip_to;
};

kernel void lstm_forward_temporal(const device bc::Buffer* input [[ buffer(0) ]],
                         const device bc::Buffer* weights [[ buffer(1) ]],
                         const device bc::Buffer* biases [[ buffer(2) ]],
                         device bc::Buffer* output [[ buffer(3) ]],
                         device bc::Buffer* activations [[ buffer(4) ]],
                         const device bc::Buffer* previous_state [[ buffer(5) ]],
                         device bc::Buffer* state [[ buffer(6) ]],
                         const device uint* time [[ buffer(7) ]],
                         constant LSTMParameters& params [[ buffer(8) ]],
                         uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(output, id))
        return;

    const auto unit = id[2];
    const auto batch_element = id[0];

    const auto input_gate_index       = 0 * params.unit_count + unit;
    const auto input_activation_index = 1 * params.unit_count + unit;
    const auto forget_gate_index      = 2 * params.unit_count + unit;
    const auto output_gate_index      = 3 * params.unit_count + unit;

    auto input_gate       = at(biases, input_gate_index);
    auto input_activation = at(biases, input_activation_index);
    auto forget_gate      = at(biases, forget_gate_index);
    auto output_gate      = at(biases, output_gate_index);

    for (uint i = 0; i < params.input_size; i += 1) {
        const auto input_value = at(input, *time * params.batch_size * params.input_size + batch_element + i * params.batch_size);
        input_gate       += at(weights, input_gate_index       + i * 4 * params.unit_count) * input_value;
        input_activation += at(weights, input_activation_index + i * 4 * params.unit_count) * input_value;
        forget_gate      += at(weights, forget_gate_index      + i * 4 * params.unit_count) * input_value;
        output_gate      += at(weights, output_gate_index      + i * 4 * params.unit_count) * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = at(previous_state, params.unit_count * (1 + batch_element * 2) + i);
        input_gate       += at(weights, input_gate_index       + j * 4 * params.unit_count) * old_out;
        input_activation += at(weights, input_activation_index + j * 4 * params.unit_count) * old_out;
        forget_gate      += at(weights, forget_gate_index      + j * 4 * params.unit_count) * old_out;
        output_gate      += at(weights, output_gate_index      + j * 4 * params.unit_count) * old_out;
    }

    const auto old_activation = at(previous_state, unit + batch_element * 2 * params.unit_count);
    auto activation = bc::sigmoid(forget_gate) * old_activation + bc::sigmoid(input_gate) * bc::tanh(input_activation);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = bc::sigmoid(output_gate) * bc::tanh(activation);

    at(output, id) = out;
    at(state, unit + batch_element * 2 * params.unit_count) = activation;
    at(state, unit + params.unit_count * (1 + batch_element * 2)) = out;

    at(activations, input_gate_index       + batch_element * 4 * params.unit_count) = input_gate;
    at(activations, input_activation_index + batch_element * 4 * params.unit_count) = input_activation;
    at(activations, forget_gate_index      + batch_element * 4 * params.unit_count) = forget_gate;
    at(activations, output_gate_index      + batch_element * 4 * params.unit_count) = output_gate;
}

kernel void lstm_forward_simple(const device bc::Buffer* input [[ buffer(0) ]],
                                const device bc::Buffer* weights [[ buffer(1) ]],
                                const device bc::Buffer* biases [[ buffer(2) ]],
                                device bc::Buffer* output [[ buffer(3) ]],
                                const device bc::Buffer* previous_state [[ buffer(4) ]],
                                device bc::Buffer* state [[ buffer(5) ]],
                                constant LSTMParameters& params [[ buffer(6) ]],
                                uint3 id [[ thread_position_in_grid ]])
{
    const auto unit = id[2];
    const auto batch_element = id[0];

    if (!isValid(output, id))
        return;

    const auto input_gate_index       = 0 * params.unit_count + unit;
    const auto input_activation_index = 1 * params.unit_count + unit;
    const auto forget_gate_index      = 2 * params.unit_count + unit;
    const auto output_gate_index      = 3 * params.unit_count + unit;

    auto input_gate       = at(biases, input_gate_index);
    auto input_activation = at(biases, input_activation_index);
    auto forget_gate      = at(biases, forget_gate_index);
    auto output_gate      = at(biases, output_gate_index);

    for (uint i = 0; i < params.input_size; i += 1) {
        const auto input_value = at(input, batch_element + i * params.batch_size);
        input_gate       += at(weights, input_gate_index       + i * 4 * params.unit_count) * input_value;
        input_activation += at(weights, input_activation_index + i * 4 * params.unit_count) * input_value;
        forget_gate      += at(weights, forget_gate_index      + i * 4 * params.unit_count) * input_value;
        output_gate      += at(weights, output_gate_index      + i * 4 * params.unit_count) * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = at(previous_state, params.unit_count * (1 + batch_element * 2) + i);
        input_gate       += at(weights, input_gate_index       + j * 4 * params.unit_count) * old_out;
        input_activation += at(weights, input_activation_index + j * 4 * params.unit_count) * old_out;
        forget_gate      += at(weights, forget_gate_index      + j * 4 * params.unit_count) * old_out;
        output_gate      += at(weights, output_gate_index      + j * 4 * params.unit_count) * old_out;
    }

    const auto old_activation = at(previous_state, unit + batch_element * 2 * params.unit_count);
    auto activation = bc::sigmoid(forget_gate) * old_activation + bc::sigmoid(input_gate) * bc::tanh(input_activation);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = bc::sigmoid(output_gate) * bc::tanh(activation);

    at(output, batch_element + unit * params.batch_size) = out;
    at(state, unit + batch_element * 2 * params.unit_count) = activation;
    at(state, unit + params.unit_count * (1 + batch_element * 2)) = out;
}

kernel void lstm_backward_activations(const device bc::Buffer* output_delta [[ buffer(0) ]],
                                      const device bc::Buffer* weights [[ buffer(1) ]],
                                      const device bc::Buffer* activations [[ buffer(2) ]],
                                      const device bc::Buffer* future_activations [[ buffer(3) ]],
                                      device bc::Buffer* activation_delta [[ buffer(4) ]],
                                      const device bc::Buffer* future_activation_delta [[ buffer(5) ]],
                                      const device bc::Buffer* state [[ buffer(6) ]],
                                      device bc::Buffer* state_delta [[ buffer(7) ]],
                                      const device bc::Buffer* old_state [[ buffer(8) ]],
                                      const device bc::Buffer* future_state_delta [[ buffer(9) ]],
                                      const device uint* time [[ buffer(10) ]],
                                      constant LSTMParameters& params [[ buffer(11) ]],
                                      uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(activation_delta, id))
        return;

    const auto unit = id[2];
    const auto batch_element = id[0];

    const auto input_gate_index        = 0 * params.unit_count + unit;
    const auto input_activation_index  = 1 * params.unit_count + unit;
    const auto forget_gate_index       = 2 * params.unit_count + unit;
    const auto output_gate_index       = 3 * params.unit_count + unit;

    // (T + 1)'s activation deltas
    const auto future_input_gate_delta       = at(future_activation_delta, batch_element * 4 * params.unit_count + input_gate_index);
    const auto future_input_activation_delta = at(future_activation_delta, batch_element * 4 * params.unit_count + input_activation_index);
    const auto future_forget_gate_delta      = at(future_activation_delta, batch_element * 4 * params.unit_count + forget_gate_index);
    const auto future_output_gate_delta      = at(future_activation_delta, batch_element * 4 * params.unit_count + output_gate_index);

    // Unrolled output delta = output_delta + (unit_weights ⨉ future_activation_deltas)
    auto true_ouput_delta = at(output_delta, *time * params.batch_size * params.unit_count + batch_element + unit * params.batch_size);
    true_ouput_delta     += at(weights, input_gate_index       + (params.input_size + unit) * 4 * params.unit_count) * future_input_gate_delta;
    true_ouput_delta     += at(weights, input_activation_index + (params.input_size + unit) * 4 * params.unit_count) * future_input_activation_delta;
    true_ouput_delta     += at(weights, forget_gate_index      + (params.input_size + unit) * 4 * params.unit_count) * future_forget_gate_delta;
    true_ouput_delta     += at(weights, output_gate_index      + (params.input_size + unit) * 4 * params.unit_count) * future_output_gate_delta;

    // T's unsmoothed activations
    const auto unsmoothed_input_gate       = at(activations, batch_element * 4 * params.unit_count + input_gate_index);
    const auto unsmoothed_input_activation = at(activations, batch_element * 4 * params.unit_count + input_activation_index);
    const auto unsmoothed_forget_gate      = at(activations, batch_element * 4 * params.unit_count + forget_gate_index);
    const auto unsmoothed_output_gate      = at(activations, batch_element * 4 * params.unit_count + output_gate_index);

    // T's smoothed activations
    const auto smoothed_input_gate       = bc::sigmoid(unsmoothed_input_gate);
    const auto smoothed_input_activation = bc::tanh(unsmoothed_input_activation);
    const auto smoothed_forget_gate      = bc::sigmoid(unsmoothed_forget_gate);
    const auto smoothed_output_gate      = bc::sigmoid(unsmoothed_output_gate);

    // State delta
    /*
     d_state = true_output_delta ⊙ smooth_output_gate ⊙ (1 − tanh^2(state)) + d_future_state ⊙ smoothed_future_forget_gate
     */
    at(state_delta, unit + batch_element * 2 * params.unit_count) = true_ouput_delta * smoothed_output_gate * (1 - bc::tanh(at(state, unit + batch_element * 2 * params.unit_count)) * bc::tanh(at(state, unit + batch_element * 2 * params.unit_count))) + at(future_state_delta, unit + batch_element * 2 * params.unit_count) * bc::sigmoid(at(future_activations, batch_element * 4 * params.unit_count + forget_gate_index));

    // Delta of unsmoothed functions
    /*
     d_unsmoothed_i = d_state       ⊙ smooth_input_activation
     d_unsmoothed_a = d_state       ⊙ smooth_input_gate
     d_unsmoothed_f = d_state       ⊙ old_state
     d_unsmoothed_o = d_true_output ⊙ tanh(state)
     */
    const auto unsmoothed_input_gate_delta       = at(state_delta, unit + batch_element * 2 * params.unit_count) * smoothed_input_activation;
    const auto unsmoothed_input_activation_delta = at(state_delta, unit + batch_element * 2 * params.unit_count) * smoothed_input_gate;
    const auto unsmoothed_forget_gate_delta      = at(state_delta, unit + batch_element * 2 * params.unit_count) * at(old_state, unit + batch_element * 2 * params.unit_count);
    const auto unsmoothed_output_gate_delta      = true_ouput_delta * bc::tanh(at(state, unit + batch_element * 2 * params.unit_count));

    // Delta of smoothed functions
    /*
     d_smoothed_i = d_smoothed_i   ⊙ smoothed_input_gate  ⊙ (1 - smoothed_input_gate)
     d_smoothed_a = d_unsmoothed_a ⊙ (1 - (smoothed_input_activation)^2)
     d_smoothed_f = d_smoothed_f   ⊙ smoothed_forget_gate ⊙ (1 - smoothed_forget_gate)
     d_smoothed_o = d_smoothed_o   ⊙ smoothed_output_gate ⊙ (1 - smoothed_output_gate)
     */
    at(activation_delta, batch_element * 4 * params.unit_count + input_gate_index)       = unsmoothed_input_gate_delta       * smoothed_input_gate  * (1 - smoothed_input_gate);
    at(activation_delta, batch_element * 4 * params.unit_count + input_activation_index) = unsmoothed_input_activation_delta * (1 - smoothed_input_activation * smoothed_input_activation);
    at(activation_delta, batch_element * 4 * params.unit_count + forget_gate_index)      = unsmoothed_forget_gate_delta      * smoothed_forget_gate * (1 - smoothed_forget_gate);
    at(activation_delta, batch_element * 4 * params.unit_count + output_gate_index)      = unsmoothed_output_gate_delta      * smoothed_output_gate * (1 - smoothed_output_gate);

    // Delta of previous input
    at(state_delta, unit + params.unit_count + batch_element * 2 * params.unit_count)  = at(weights, input_gate_index + (params.input_size + unit) * 4 * params.unit_count)       * at(activation_delta, batch_element * 4 * params.unit_count + input_gate_index);
    at(state_delta, unit + params.unit_count + batch_element * 2 * params.unit_count) += at(weights, input_activation_index + (params.input_size + unit) * 4 * params.unit_count) * at(activation_delta, batch_element * 4 * params.unit_count + input_activation_index);
    at(state_delta, unit + params.unit_count + batch_element * 2 * params.unit_count) += at(weights, forget_gate_index + (params.input_size + unit) * 4 * params.unit_count)      * at(activation_delta, batch_element * 4 * params.unit_count + forget_gate_index);
    at(state_delta, unit + params.unit_count + batch_element * 2 * params.unit_count) += at(weights, output_gate_index + (params.input_size + unit) * 4 * params.unit_count)      * at(activation_delta, batch_element * 4 * params.unit_count + output_gate_index);
}

kernel void lstm_backward_weights(const device bc::Buffer* input [[ buffer(0) ]],
                                  const device bc::Buffer* state [[ buffer(1) ]],
                                  device bc::Buffer* weights_delta [[ buffer(2) ]],
                                  device bc::Buffer* bias_delta [[ buffer(3) ]],
                                  const device bc::Buffer* activation_delta [[ buffer(4) ]],
                                  const device bc::Buffer* future_activation_delta [[ buffer(5) ]],
                                  const device uint* time [[ buffer(6) ]],
                                  constant LSTMParameters& params [[ buffer(7) ]],
                                  uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(weights_delta, id))
        return;

    const auto unit = id[0];

    const auto input_gate_index       = 0 * params.unit_count + unit;
    const auto input_activation_index = 1 * params.unit_count + unit;
    const auto forget_gate_index      = 2 * params.unit_count + unit;
    const auto output_gate_index      = 3 * params.unit_count + unit;


    for (auto batch_element = 0; batch_element < params.batch_size; batch_element += 1) {
        for (auto input_element = 0; input_element < params.input_size; input_element += 1) {
            const auto input_value = at(input, *time * params.batch_size * params.input_size + batch_element + input_element * params.batch_size);
            at(weights_delta, input_gate_index       + input_element * 4 * params.unit_count) += at(activation_delta, batch_element * 4 * params.unit_count + input_gate_index)       * input_value;
            at(weights_delta, input_activation_index + input_element * 4 * params.unit_count) += at(activation_delta, batch_element * 4 * params.unit_count + input_activation_index) * input_value;
            at(weights_delta, forget_gate_index      + input_element * 4 * params.unit_count) += at(activation_delta, batch_element * 4 * params.unit_count + forget_gate_index)      * input_value;
            at(weights_delta, output_gate_index      + input_element * 4 * params.unit_count) += at(activation_delta, batch_element * 4 * params.unit_count + output_gate_index)      * input_value;
        }
        for (auto unit = 0; unit < params.unit_count; unit += 1) {
            const auto j = unit + params.input_size;
            const auto old_out = at(state, params.unit_count * (1 + batch_element * 2) + unit);
            at(weights_delta, input_gate_index       + j * 4 * params.unit_count) += at(future_activation_delta, batch_element * 4 * params.unit_count + input_gate_index)       * old_out;
            at(weights_delta, input_activation_index + j * 4 * params.unit_count) += at(future_activation_delta, batch_element * 4 * params.unit_count + input_activation_index) * old_out;
            at(weights_delta, forget_gate_index      + j * 4 * params.unit_count) += at(future_activation_delta, batch_element * 4 * params.unit_count + forget_gate_index)      * old_out;
            at(weights_delta, output_gate_index      + j * 4 * params.unit_count) += at(future_activation_delta, batch_element * 4 * params.unit_count + output_gate_index)      * old_out;

        }
        at(bias_delta, input_gate_index)       = at(activation_delta, batch_element * 4 * params.unit_count + input_gate_index);
        at(bias_delta, input_activation_index) = at(activation_delta, batch_element * 4 * params.unit_count + input_activation_index);
        at(bias_delta, forget_gate_index)      = at(activation_delta, batch_element * 4 * params.unit_count + forget_gate_index);
        at(bias_delta, output_gate_index)      = at(activation_delta, batch_element * 4 * params.unit_count + output_gate_index);
    }
}

kernel void lstm_backward_inputs(device bc::Buffer* input_delta [[ buffer(0) ]],
                                 const device bc::Buffer* weights [[ buffer(1) ]],
                                 const device bc::Buffer* activation_delta [[ buffer(2) ]],
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

        at(input_delta, *time * params.batch_size * params.input_size + element_index)  = at(weights, input_gate_index + input_element * 4 * params.unit_count)       * at(activation_delta, batch_element * 4 * params.unit_count + input_gate_index);
        at(input_delta, *time * params.batch_size * params.input_size + element_index) += at(weights, input_activation_index + input_element * 4 * params.unit_count) * at(activation_delta, batch_element * 4 * params.unit_count + input_activation_index);
        at(input_delta, *time * params.batch_size * params.input_size + element_index) += at(weights, forget_gate_index + input_element * 4 * params.unit_count)      * at(activation_delta, batch_element * 4 * params.unit_count + forget_gate_index);
        at(input_delta, *time * params.batch_size * params.input_size + element_index) += at(weights, output_gate_index + input_element * 4 * params.unit_count)      * at(activation_delta, batch_element * 4 * params.unit_count + output_gate_index);
    }
}
