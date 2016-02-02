// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

public typealias Blob = ValueArray<Float>

public protocol Layer {
}

public protocol DataLayer : Layer {
    var data: Blob { get }
}

public protocol ForwardLayer : Layer {
    var outputSize: Int { get }

    /// Forward-propagate the input
    func encodeForwardInBuffer(buffer: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer)
}

public protocol BackwardLayer : Layer {
    /// Backward-propagate the output differences
    func encodeBackwardInBuffer(buffer: MTLCommandBuffer, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer)
}

public protocol SinkLayer : Layer {
    func consume(input: Blob)
}
