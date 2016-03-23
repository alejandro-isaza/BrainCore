// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;


struct SolverParameters {
    float learningRate;
    float momentum;
};

struct ParameterDimensions {
    uint output_size;
};

kernel void sgd_update_parameters(device float* parameter [[ buffer(0) ]],
                              const device float* parameterDiff [[ buffer(1) ]],
                              constant SolverParameters& solverParams [[ buffer(2) ]],
                              constant ParameterDimensions& dims [[ buffer(3) ]],
                              uint id [[ thread_position_in_grid ]])
{
    if (id >= dims.output_size)
        return;
    
    parameter[id] = solverParams.momentum * parameter[id] - solverParams.learningRate * parameterDiff[id];
}
