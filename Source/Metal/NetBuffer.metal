// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

#include <metal_stdlib>
#include <metal_common>

using namespace metal;


kernel void reset_buffer(device float* input [[ buffer(0) ]],
                         constant uint* count [[ buffer(1) ]],
                         uint elementIndex [[ thread_position_in_grid ]])
{
    if (elementIndex >= *count) {
        return;
    }
    
    input[elementIndex] = 0.0;
}
