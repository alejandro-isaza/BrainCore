// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

#include "Utilities.h"

using namespace metal;


struct SolverParameters {
    float learningRate;
};

kernel void sgd_update_parameters(device bc::Buffer* parameter [[ buffer(0) ]],
                                  const device bc::Buffer* parameterDiff [[ buffer(1) ]],
                                  constant SolverParameters& solverParams [[ buffer(2) ]],
                                  constant uint* output_size [[ buffer(3) ]],
                                  uint id [[ thread_position_in_grid ]])
{
    if (id >= *output_size)
        return;
    
    parameter[id] = parameter[id] - solverParams.learningRate * parameterDiff[id];
}
