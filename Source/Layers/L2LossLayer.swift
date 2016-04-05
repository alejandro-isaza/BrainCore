// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class L2LossLayer: LossLayer {
    public var loss: Double = 0.0
    
    public let size: Int
    public var forwardState: MTLComputePipelineState!
    public var backwardState: MTLComputePipelineState!
    
    public var dimensionsBuffer: MTLBuffer!
    
    public var outputSize: Int {
        return 1
    }
    public var inputSize: Int {
        return size
    }
    
    struct L2LossDimensions {
        let inputSize: UInt16
        let batchSize: UInt16
    }
    
    public init(size: Int) {
        self.size = size
    }
    
    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardFunction = library.newFunctionWithName("l2_loss_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)
        
        let backwardFunction = library.newFunctionWithName("l2_loss_backward")!
        backwardState = try library.device.newComputePipelineStateWithFunction(backwardFunction)
    }
    
    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        var dimensions = L2LossDimensions(inputSize: UInt16(inputSize), batchSize: UInt16(batchSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(L2LossDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "L2LossDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "L2LossForward"
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 1)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 2)
        
        let count = batchSize / sizeof(Float)
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: 1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        encoder.endEncoding()
    }
    
    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        encodeBackwardInBuffer(buffer, batchSize: batchSize, input: input, inputDiff: inputDiff)
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, inputDiff: MTLBuffer) {
        var dimensions = L2LossDimensions(inputSize: UInt16(inputSize), batchSize: UInt16(batchSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(L2LossDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "L2LossDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "L2LossBackward"
        encoder.setComputePipelineState(backwardState)
        encoder.setBuffer(input, offset: 0, atIndex: 0)
        encoder.setBuffer(inputDiff, offset: 0, atIndex: 1)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 2)

        let count = inputSize / (2 * sizeof(Float))
        let threadsPerGroup = MTLSize(width: backwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / backwardState.threadExecutionWidth + 1, height: batchSize, depth: 1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        encoder.endEncoding()
    }
}
