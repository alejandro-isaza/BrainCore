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

    public var dimensionsBuffer: MTLBuffer!

    public var outputSize: Int {
        return size
    }

    public var inputSize: Int {
        return size
    }

    public init(size: Int) {
        self.size = size
    }
    
    struct ReluDimensions {
        let batchSize: UInt32
        let size: UInt32
    }

    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardFunction = library.newFunctionWithName("linear_rectify_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)

        let backwardFunction = library.newFunctionWithName("linear_rectify_backward")!
        backwardState = try library.device.newComputePipelineStateWithFunction(backwardFunction)
    }

    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        var dimensions = ReluDimensions(batchSize: UInt32(batchSize), size: UInt32(size))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(ReluDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "ReluDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "ReLUForward"
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 1)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 2)

        let count = input.length / sizeof(Float)
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        var dimensions = ReluDimensions(batchSize: UInt32(batchSize), size: UInt32(size))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(ReluDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "ReluDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "ReLUBackward"
        encoder.setComputePipelineState(backwardState)
        encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
        encoder.setBuffer(input, offset: 0, atIndex: 1)
        encoder.setBuffer(inputDiff, offset: 0, atIndex: 2)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 3)

        let count = outputDiff.length / sizeof(Float)
        let threadsPerGroup = MTLSize(width: backwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / backwardState.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}
