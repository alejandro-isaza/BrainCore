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
    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>

    public var inputSize: Int {
        return weights.rows
    }

    public var outputSize: Int {
        return weights.columns
    }

    public var forwardState: MTLComputePipelineState!
    public var backwardParamsState: MTLComputePipelineState!
    public var backwardInputState: MTLComputePipelineState!

    public var weightsBuffer: MTLBuffer!
    public var biasesBuffer: MTLBuffer!
    public var dimensionsBuffer: MTLBuffer!

    public var weightDiff: MTLBuffer?
    public var biasDiff: MTLBuffer?

    struct InnerProductDimensions {
        let batchSize: UInt16
        let inputSize: UInt16
        let outputSize: UInt16
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>) {
        self.weights = weights
        self.biases = biases
        precondition(biases.count == outputSize)
    }

    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardFunction = library.newFunctionWithName("inner_product_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)

        let backwardParamsFunction = library.newFunctionWithName("inner_product_backward_params")!
        backwardParamsState = try library.device.newComputePipelineStateWithFunction(backwardParamsFunction)

        let backwardInputFunction = library.newFunctionWithName("inner_product_backward_input")!
        backwardInputState = try library.device.newComputePipelineStateWithFunction(backwardInputFunction)

        withPointer(weights) { pointer in
            weightsBuffer = library.device.newBufferWithBytes(pointer, length: inputSize * outputSize * sizeof(Float), options: .CPUCacheModeWriteCombined)
        }
        weightsBuffer.label = "InnerProductWeights"

        withPointer(biases) { pointer in
            biasesBuffer = library.device.newBufferWithBytes(pointer, length: outputSize * sizeof(Float), options: .CPUCacheModeWriteCombined)
        }
        biasesBuffer.label = "InnerProductBiases"
    }

    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        var dimensions = InnerProductDimensions(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(InnerProductDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "InnerProductDimensions"

        
        let encoder = buffer.computeCommandEncoder()
        encoder.label = "InnerProductForward"
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 3)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 4)
        
        let count = outputSize
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        var dimensions = InnerProductDimensions(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(InnerProductDimensions), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "InnerProductDimensions"

        
        if weightDiff == nil {
            weightDiff = buffer.device.newBufferWithLength(inputSize * outputSize, options: .CPUCacheModeDefaultCache)
            weightDiff!.label = "InnerProductWeightDiffs"
        }
        if biasDiff == nil {
            biasDiff = buffer.device.newBufferWithLength(outputSize, options: .CPUCacheModeDefaultCache)
            biasDiff!.label = "InnerProductBiasDiffs"
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.label = "InnerProductBackwardParams"
            encoder.setComputePipelineState(backwardParamsState)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(input, offset: 0, atIndex: 1)
            encoder.setBuffer(weightDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(biasDiff, offset: 0, atIndex: 3)
            encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 4)

            let count = outputSize
            let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: 1, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.label = "InnerProductBackwardState"
            encoder.setComputePipelineState(backwardInputState)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
            encoder.setBuffer(inputDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 3)

            let count = inputSize
            let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: batchSize, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }
    }
}
