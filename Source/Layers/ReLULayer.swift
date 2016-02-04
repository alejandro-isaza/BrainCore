// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class ReLULayer: ForwardLayer, BackwardLayer {
    public let size: Int
    public var forwardState: MTLComputePipelineState!
    public var backwardState: MTLComputePipelineState!

    public var outputSize: Int {
        return size
    }

    public var inputSize: Int {
        return size
    }

    public init(net: Net, size: Int) throws {
        self.size = size

        let library = net.library
        let forwardFunction = library.newFunctionWithName("linear_rectify_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)

        let backwardFunction = library.newFunctionWithName("linear_rectify_backward")!
        backwardState = try library.device.newComputePipelineStateWithFunction(backwardFunction)
    }

    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer) {
        let encoder = buffer.computeCommandEncoder()
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: 0, atIndex: 0)
        encoder.setBuffer(output, offset: 0, atIndex: 1)

        let count = input.length / sizeof(Float)
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height:1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        let encoder = buffer.computeCommandEncoder()
        encoder.setComputePipelineState(backwardState)
        encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
        encoder.setBuffer(input, offset: 0, atIndex: 1)
        encoder.setBuffer(inputDiff, offset: 0, atIndex: 2)

        let count = outputDiff.length / sizeof(Float)
        let threadsPerGroup = MTLSize(width: backwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / backwardState.threadExecutionWidth + 1, height:1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}
