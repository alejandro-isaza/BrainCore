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

kernel void lstm_forward_temporal(const device Buffer* input [[ buffer(0) ]],
                         const device Buffer* weights [[ buffer(1) ]],
                         const device Buffer* biases [[ buffer(2) ]],
                         device Buffer* output [[ buffer(3) ]],
                         device Buffer* activations [[ buffer(4) ]],
                         const device Buffer* previous_state [[ buffer(5) ]],
                         device Buffer* state [[ buffer(6) ]],
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
        const auto input_value = at(input, {i, *time, batch_element});
        input_gate       += at(weights, {i, input_gate_index}) * input_value;
        input_activation += at(weights, {i, input_activation_index}) * input_value;
        forget_gate      += at(weights, {i, forget_gate_index}) * input_value;
        output_gate      += at(weights, {i, output_gate_index}) * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = at(previous_state, {batch_element, params.unit_count + i});
        input_gate       += at(weights, {j, input_gate_index}) * old_out;
        input_activation += at(weights, {j, input_activation_index}) * old_out;
        forget_gate      += at(weights, {j, forget_gate_index}) * old_out;
        output_gate      += at(weights, {j, output_gate_index}) * old_out;
    }

    const auto old_activation = at(previous_state, {batch_element, unit});
    auto activation = sigmoid(forget_gate) * old_activation + sigmoid(input_gate) * bc::tanh(input_activation);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = sigmoid(output_gate) * bc::tanh(activation);

    at(output, id) = out;
    at(state, {batch_element, unit}) = activation;
    at(state, {batch_element, params.unit_count + unit}) = out;

    at(activations, {batch_element, input_gate_index}) = input_gate;
    at(activations, {batch_element, input_activation_index}) = input_activation;
    at(activations, {batch_element, forget_gate_index}) = forget_gate;
    at(activations, {batch_element, output_gate_index}) = output_gate;
}

kernel void lstm_forward_simple(const device Buffer* input [[ buffer(0) ]],
                                const device Buffer* weights [[ buffer(1) ]],
                                const device Buffer* biases [[ buffer(2) ]],
                                device Buffer* output [[ buffer(3) ]],
                                const device Buffer* previous_state [[ buffer(4) ]],
                                device Buffer* state [[ buffer(5) ]],
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
        const auto input_value = at(input, {i, batch_element, 0});
        input_gate       += at(weights, {i, input_gate_index}) * input_value;
        input_activation += at(weights, {i, input_activation_index}) * input_value;
        forget_gate      += at(weights, {i, forget_gate_index}) * input_value;
        output_gate      += at(weights, {i, output_gate_index}) * input_value;
    }
    for (uint i = 0; i < params.unit_count; i += 1) {
        const auto j = i + params.input_size;
        const auto old_out = at(previous_state, {batch_element, params.unit_count + i});
        input_gate       += at(weights, {j, input_gate_index}) * old_out;
        input_activation += at(weights, {j, input_activation_index}) * old_out;
        forget_gate      += at(weights, {j, forget_gate_index}) * old_out;
        output_gate      += at(weights, {j, output_gate_index}) * old_out;
    }

    const auto old_activation = at(previous_state, {batch_element, unit});
    auto activation = sigmoid(forget_gate) * old_activation + sigmoid(input_gate) * bc::tanh(input_activation);
    if (params.clip_to > 0) {
        activation = clamp(activation, -params.clip_to, params.clip_to);
    }
    const auto out = sigmoid(output_gate) * bc::tanh(activation);

    at(output, id) = out;
    at(state, {batch_element, unit}) = activation;
    at(state, {batch_element, params.unit_count + unit}) = out;
}

kernel void lstm_backward_activations(const device Buffer* output_delta [[ buffer(0) ]],
                                      const device Buffer* weights [[ buffer(1) ]],
                                      const device Buffer* activations [[ buffer(2) ]],
                                      const device Buffer* future_activations [[ buffer(3) ]],
                                      device Buffer* activation_delta [[ buffer(4) ]],
                                      const device Buffer* future_activation_delta [[ buffer(5) ]],
                                      const device Buffer* state [[ buffer(6) ]],
                                      device Buffer* state_delta [[ buffer(7) ]],
                                      const device Buffer* old_state [[ buffer(8) ]],
                                      const device Buffer* future_state_delta [[ buffer(9) ]],
                                      const device uint* time [[ buffer(10) ]],
                                      constant LSTMParameters& params [[ buffer(11) ]],
                                      uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(activation_delta, id))
        return;

    const auto unit = id[0];
    const auto batch_element = id[1];

    const auto input_gate_index        = 0 * params.unit_count + unit;
    const auto input_activation_index  = 1 * params.unit_count + unit;
    const auto forget_gate_index       = 2 * params.unit_count + unit;
    const auto output_gate_index       = 3 * params.unit_count + unit;

    // (T + 1)'s activation deltas
    const auto future_input_gate_delta       = at(future_activation_delta, {batch_element, input_gate_index});
    const auto future_input_activation_delta = at(future_activation_delta, {batch_element, input_activation_index});
    const auto future_forget_gate_delta      = at(future_activation_delta, {batch_element, forget_gate_index});
    const auto future_output_gate_delta      = at(future_activation_delta, {batch_element, output_gate_index});

    // Unrolled output delta = output_delta + (unit_weights ⨉ future_activation_deltas)
    auto true_ouput_delta = at(output_delta, *time * params.batch_size * params.unit_count + batch_element + unit * params.batch_size);
    true_ouput_delta     += at(weights, {params.input_size + unit, input_gate_index}) * future_input_gate_delta;
    true_ouput_delta     += at(weights, {params.input_size + unit, input_activation_index}) * future_input_activation_delta;
    true_ouput_delta     += at(weights, {params.input_size + unit, forget_gate_index}) * future_forget_gate_delta;
    true_ouput_delta     += at(weights, {params.input_size + unit, output_gate_index}) * future_output_gate_delta;

    // T's unsmoothed activations
    const auto unsmoothed_input_gate       = at(activations, {batch_element, input_gate_index});
    const auto unsmoothed_input_activation = at(activations, {batch_element, input_activation_index});
    const auto unsmoothed_forget_gate      = at(activations, {batch_element, forget_gate_index});
    const auto unsmoothed_output_gate      = at(activations, {batch_element, output_gate_index});

    // T's smoothed activations
    const auto smoothed_input_gate       = sigmoid(unsmoothed_input_gate);
    const auto smoothed_input_activation = bc::tanh(unsmoothed_input_activation);
    const auto smoothed_forget_gate      = sigmoid(unsmoothed_forget_gate);
    const auto smoothed_output_gate      = sigmoid(unsmoothed_output_gate);

    // State delta
    /*
     d_state = true_output_delta ⊙ smooth_output_gate ⊙ (1 − tanh^2(state)) + d_future_state ⊙ smoothed_future_forget_gate
     */
    at(state_delta, {batch_element, unit}) = true_ouput_delta * smoothed_output_gate * (1 - bc::tanh(at(state, unit + batch_element * 2 * params.unit_count)) * bc::tanh(at(state, unit + batch_element * 2 * params.unit_count))) + at(future_state_delta, {batch_element, unit}) * sigmoid(at(future_activations, {batch_element, forget_gate_index}));

    // Delta of unsmoothed functions
    /*
     d_unsmoothed_i = d_state       ⊙ smooth_input_activation
     d_unsmoothed_a = d_state       ⊙ smooth_input_gate
     d_unsmoothed_f = d_state       ⊙ old_state
     d_unsmoothed_o = d_true_output ⊙ tanh(state)
     */
    const auto unsmoothed_input_gate_delta       = at(state_delta, {batch_element, unit}) * smoothed_input_activation;
    const auto unsmoothed_input_activation_delta = at(state_delta, {batch_element, unit}) * smoothed_input_gate;
    const auto unsmoothed_forget_gate_delta      = at(state_delta, {batch_element, unit}) * at(old_state, {batch_element, unit});
    const auto unsmoothed_output_gate_delta      = true_ouput_delta * bc::tanh(at(state, {batch_element, unit}));

    // Delta of smoothed functions
    /*
     d_smoothed_i = d_smoothed_i   ⊙ smoothed_input_gate  ⊙ (1 - smoothed_input_gate)
     d_smoothed_a = d_unsmoothed_a ⊙ (1 - (smoothed_input_activation)^2)
     d_smoothed_f = d_smoothed_f   ⊙ smoothed_forget_gate ⊙ (1 - smoothed_forget_gate)
     d_smoothed_o = d_smoothed_o   ⊙ smoothed_output_gate ⊙ (1 - smoothed_output_gate)
     */
    at(activation_delta, {batch_element, input_gate_index})       = unsmoothed_input_gate_delta       * smoothed_input_gate  * (1 - smoothed_input_gate);
    at(activation_delta, {batch_element, input_activation_index}) = unsmoothed_input_activation_delta * (1 - smoothed_input_activation * smoothed_input_activation);
    at(activation_delta, {batch_element, forget_gate_index})      = unsmoothed_forget_gate_delta      * smoothed_forget_gate * (1 - smoothed_forget_gate);
    at(activation_delta, {batch_element, output_gate_index})      = unsmoothed_output_gate_delta      * smoothed_output_gate * (1 - smoothed_output_gate);

    // Delta of previous input
    at(state_delta, {batch_element, params.unit_count + unit})  = at(weights, {params.input_size + unit, input_gate_index})       * at(activation_delta, {batch_element, input_gate_index});
    at(state_delta, {batch_element, params.unit_count + unit}) += at(weights, {params.input_size + unit, input_activation_index}) * at(activation_delta, {batch_element, input_gate_index});
    at(state_delta, {batch_element, params.unit_count + unit}) += at(weights, {params.input_size + unit, forget_gate_index})      * at(activation_delta, {batch_element, input_gate_index});
    at(state_delta, {batch_element, params.unit_count + unit}) += at(weights, {params.input_size + unit, output_gate_index})      * at(activation_delta, {batch_element, input_gate_index});
}

kernel void lstm_backward_weights(const device Buffer* input [[ buffer(0) ]],
                                  const device Buffer* state [[ buffer(1) ]],
                                  device Buffer* weights_delta [[ buffer(2) ]],
                                  device Buffer* bias_delta [[ buffer(3) ]],
                                  const device Buffer* activation_delta [[ buffer(4) ]],
                                  const device Buffer* future_activation_delta [[ buffer(5) ]],
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


    for (unsigned int batch_element = 0; batch_element < params.batch_size; batch_element += 1) {
        for (unsigned int input_element = 0; input_element < params.input_size; input_element += 1) {
            const auto input_value = at(input, {input_element, *time, batch_element});
            at(weights_delta, {input_element, input_gate_index}) += at(activation_delta, {batch_element, input_gate_index})       * input_value;
            at(weights_delta, {input_element, input_activation_index}) += at(activation_delta, {batch_element, input_activation_index}) * input_value;
            at(weights_delta, {input_element, forget_gate_index}) += at(activation_delta, {batch_element, forget_gate_index})      * input_value;
            at(weights_delta, {input_element, output_gate_index}) += at(activation_delta, {batch_element, output_gate_index})      * input_value;
        }
        for (unsigned int unit = 0; unit < params.unit_count; unit += 1) {
            const auto j = unit + params.input_size;
            const auto old_out = at(state, {batch_element, params.unit_count + unit});
            at(weights_delta, {j, input_gate_index}) += at(future_activation_delta, {batch_element, input_gate_index})       * old_out;
            at(weights_delta, {j, input_activation_index}) += at(future_activation_delta, {batch_element, input_activation_index}) * old_out;
            at(weights_delta, {j, forget_gate_index}) += at(future_activation_delta, {batch_element, forget_gate_index})      * old_out;
            at(weights_delta, {j, output_gate_index}) += at(future_activation_delta, {batch_element, output_gate_index})      * old_out;

        }
        at(bias_delta, input_gate_index)       = at(activation_delta, {batch_element, input_gate_index});
        at(bias_delta, input_activation_index) = at(activation_delta, {batch_element, input_activation_index});
        at(bias_delta, forget_gate_index)      = at(activation_delta, {batch_element, forget_gate_index});
        at(bias_delta, output_gate_index)      = at(activation_delta, {batch_element, output_gate_index});
    }
}

kernel void lstm_backward_inputs(device Buffer* input_delta [[ buffer(0) ]],
                                 const device Buffer* weights [[ buffer(1) ]],
                                 const device Buffer* activation_delta [[ buffer(2) ]],
                                 const device uint* time [[ buffer(3) ]],
                                 constant LSTMParameters& params [[ buffer(4) ]],
                                 uint3 id [[ thread_position_in_grid ]])
{
    if (!isValid(input_delta, id))
        return;

    const auto input_element = id[2];
    const auto batch_element = id[0];

    for (unsigned int unit = 0; unit < params.unit_count; unit += 1) {
        const auto input_gate_index       = 0 * params.unit_count + unit;
        const auto input_activation_index = 1 * params.unit_count + unit;
        const auto forget_gate_index      = 2 * params.unit_count + unit;
        const auto output_gate_index      = 3 * params.unit_count + unit;

        at(input_delta, id)  = at(weights, {input_element, input_gate_index})       * at(activation_delta, {batch_element, input_gate_index});
        at(input_delta, id) += at(weights, {input_element, input_activation_index}) * at(activation_delta, {batch_element, input_activation_index});
        at(input_delta, id) += at(weights, {input_element, forget_gate_index})      * at(activation_delta, {batch_element, forget_gate_index});
        at(input_delta, id) += at(weights, {input_element, output_gate_index})      * at(activation_delta, {batch_element, output_gate_index});
    }
}
