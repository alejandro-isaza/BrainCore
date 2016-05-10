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


kernel void reset_buffer(device bc::Buffer* input [[ buffer(0) ]],
                         uint3 elementIndex [[ thread_position_in_grid ]])
{
    at(input, elementIndex) = 0.0;
}
