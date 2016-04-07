// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class L2LossLayer: LossLayer {
    struct Parameters {
        let batchSize: UInt16
        let inputSize: UInt16
    }

    public let size: Int
    public var loss: Double = 0.0
    
    public var outputSize: Int {
        return 1
    }
    public var inputSize: Int {
        return 2 * size
    }
    
    public init(size: Int) {
        self.size = size
    }

    static let forwardFunctionName = "sigmoid_forward"
    static let backwardFunctionName = "sigmoid_backward"

    var forwardFunction: MTLComputePipelineState!
    var backwardFunction: MTLComputePipelineState!
    
    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardLibraryFunction = library.newFunctionWithName("l2_loss_forward")!
        forwardFunction = try library.device.newComputePipelineStateWithFunction(forwardLibraryFunction)
        
        let backwardLibraryFunction = library.newFunctionWithName("l2_loss_backward")!
        backwardFunction = try library.device.newComputePipelineStateWithFunction(backwardLibraryFunction)
    }
    
    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        var dimensions = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize / 2))
        let dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(Parameters), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "L2LossDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "L2LossForward"
        encoder.setComputePipelineState(forwardFunction)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 1)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 2)

        let threadsPerGroup = MTLSize(width: forwardFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (batchSize - 1) / forwardFunction.threadExecutionWidth + 1, height: 1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        encoder.endEncoding()
    }

    public func encodeBackwardLossInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, deltas: MTLBuffer) {
        var dimensions = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize / 2))
        let dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(Parameters), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "L2LossDimensions"

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "L2LossBackward"
        encoder.setComputePipelineState(backwardFunction)
        encoder.setBuffer(input, offset: 0, atIndex: 0)
        encoder.setBuffer(deltas, offset: 0, atIndex: 1)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 2)

        let threadsPerGroup = MTLSize(width: backwardFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: ((inputSize / 2) - 1) / backwardFunction.threadExecutionWidth + 1, height: batchSize, depth: 1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        encoder.endEncoding()
    }
}
