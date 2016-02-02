// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal
import Upsurge

public class InnerProductLayer: ForwardLayer, BackwardLayer {
    public let inputSize: Int
    public let outputSize: Int

    public var forwardState: MTLComputePipelineState!
    public var backwardParamsState: MTLComputePipelineState!
    public var backwardInputState: MTLComputePipelineState!

    public var weights: MTLBuffer!
    public var biases: MTLBuffer!
    public var dimensions: MTLBuffer!

    public var weightDiff: MTLBuffer?
    public var biasDiff: MTLBuffer?

    struct InnerProductDimensions {
        let inputSize: UInt16
        let outputSize: UInt16
    }

    public init<M: QuadraticType, A: LinearType where M.Element == Float, A.Element == Float>(library: MTLLibrary, weights: M, biases: A) throws {
        inputSize = weights.rows
        outputSize = weights.columns
        precondition(biases.count == outputSize)

        let forwardFunction = library.newFunctionWithName("inner_product_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)

        let backwardParamsFunction = library.newFunctionWithName("inner_product_backward_params")!
        backwardParamsState = try library.device.newComputePipelineStateWithFunction(backwardParamsFunction)

        let backwardInputFunction = library.newFunctionWithName("inner_product_backward_input")!
        backwardInputState = try library.device.newComputePipelineStateWithFunction(backwardInputFunction)

        self.weights = library.device.newBufferWithBytes(weights.pointer, length: weights.count * sizeof(Float), options: .StorageModePrivate)
        self.biases = library.device.newBufferWithBytes(biases.pointer, length: biases.count * sizeof(Float), options: .StorageModePrivate)

        var dimensions = InnerProductDimensions(inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        self.dimensions = library.device.newBufferWithBytes(&dimensions, length: sizeof(InnerProductDimensions), options: .StorageModePrivate)
    }

    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer) {
        let encoder = buffer.computeCommandEncoder()
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: 0, atIndex: 0)
        encoder.setBuffer(weights, offset: 0, atIndex: 1)
        encoder.setBuffer(biases, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: 0, atIndex: 3)
        encoder.setBuffer(dimensions, offset: 0, atIndex: 4)
        
        let count = outputSize
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height:1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        if weightDiff == nil {
            weightDiff = buffer.device.newBufferWithLength(inputSize * outputSize, options: .CPUCacheModeDefaultCache)
        }
        if biasDiff == nil {
            biasDiff = buffer.device.newBufferWithLength(outputSize, options: .CPUCacheModeDefaultCache)
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.setComputePipelineState(backwardParamsState)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(input, offset: 0, atIndex: 1)
            encoder.setBuffer(weightDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(biasDiff, offset: 0, atIndex: 3)
            encoder.setBuffer(dimensions, offset: 0, atIndex: 4)

            let count = outputSize
            let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height:1, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.setComputePipelineState(backwardInputState)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(weights, offset: 0, atIndex: 1)
            encoder.setBuffer(inputDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(dimensions, offset: 0, atIndex: 3)

            let count = inputSize
            let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height:1, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }
    }
}
