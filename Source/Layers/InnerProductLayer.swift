// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal
import Upsurge

public class InnerProductLayer: BackwardLayer, TrainableLayer {
    struct Parameters {
        let batchSize: UInt16
        let inputSize: UInt16
        let outputSize: UInt16
    }

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>

    public var inputSize: Int {
        return weights.rows
    }

    public var outputSize: Int {
        return weights.columns
    }

    public var weightsBuffer: MTLBuffer!
    public var biasesBuffer: MTLBuffer!
    public var dimensionsBuffer: MTLBuffer!

    public var weightDiff: MTLBuffer?
    public var biasDiff: MTLBuffer?

    public init(weights: Matrix<Float>, biases: ValueArray<Float>) {
        self.weights = weights
        self.biases = biases
        precondition(biases.count == outputSize)
    }

    static let forwardFunctionName = "inner_product_forward"
    static let backwardParamsFunctionName = "inner_product_backward_params"
    static let backwardInputFunctionName = "inner_product_backward_input"

    var forwardFunction: MTLComputePipelineState!
    var backwardParamsFunction: MTLComputePipelineState!
    var backwardInputFunction: MTLComputePipelineState!
    var updateFunction: MTLComputePipelineState!

    public func setupInLibrary(library: MTLLibrary, updateFunction: MTLComputePipelineState) throws {
        self.updateFunction = updateFunction
        try setupInLibrary(library)
    }
    
    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardLibraryFunction = library.newFunctionWithName(InnerProductLayer.forwardFunctionName)!
        forwardFunction = try library.device.newComputePipelineStateWithFunction(forwardLibraryFunction)

        let backwardParamsLibraryFunction = library.newFunctionWithName(InnerProductLayer.backwardParamsFunctionName)!
        backwardParamsFunction = try library.device.newComputePipelineStateWithFunction(backwardParamsLibraryFunction)

        let backwardInputLibraryFunction = library.newFunctionWithName(InnerProductLayer.backwardInputFunctionName)!
        backwardInputFunction = try library.device.newComputePipelineStateWithFunction(backwardInputLibraryFunction)

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
        var dimensions = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(Parameters), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "InnerProductDimensions"

        
        let encoder = buffer.computeCommandEncoder()
        encoder.label = "InnerProductForward"
        encoder.setComputePipelineState(forwardFunction)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 3)
        encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 4)

        let threadsPerGroup = MTLSize(width: forwardFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (outputSize - 1) / forwardFunction.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    public func encodeBackwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, outputDiff: MTLBuffer, input: MTLBuffer, inputDiff: MTLBuffer) {
        var dimensions = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        dimensionsBuffer = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(Parameters), options: .CPUCacheModeWriteCombined)
        dimensionsBuffer.label = "InnerProductDimensions"

        
        if weightDiff == nil {
            weightDiff = buffer.device.newBufferWithLength(inputSize * outputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
            weightDiff!.label = "InnerProductWeightDiffs"
        }
        if biasDiff == nil {
            biasDiff = buffer.device.newBufferWithLength(outputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
            biasDiff!.label = "InnerProductBiasDiffs"
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.label = "InnerProductBackwardParams"
            encoder.setComputePipelineState(backwardParamsFunction)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(input, offset: 0, atIndex: 1)
            encoder.setBuffer(weightDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(biasDiff, offset: 0, atIndex: 3)
            encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 4)

            let threadsPerGroup = MTLSize(width: forwardFunction.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (outputSize - 1) / forwardFunction.threadExecutionWidth + 1, height: 1, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }

        do {
            let encoder = buffer.computeCommandEncoder()
            encoder.label = "InnerProductBackwardState"
            encoder.setComputePipelineState(backwardInputFunction)
            encoder.setBuffer(outputDiff, offset: 0, atIndex: 0)
            encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
            encoder.setBuffer(inputDiff, offset: 0, atIndex: 2)
            encoder.setBuffer(dimensionsBuffer, offset: 0, atIndex: 3)

            let threadsPerGroup = MTLSize(width: forwardFunction.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (inputSize - 1) / forwardFunction.threadExecutionWidth + 1, height: batchSize, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
        }
    }

    public func update(updateParameter: (parameter: MTLBuffer, parameterDifference: MTLBuffer) -> Void) {
        guard let weightDiff = weightDiff else {
            fatalError("Inner Product weights were not initialized")
        }
        guard let biasDiff = biasDiff else {
            fatalError("Inner Product biases were not initialized")
        }

        updateParameter(parameter: weightsBuffer, parameterDifference: weightDiff)
        updateParameter(parameter: biasesBuffer, parameterDifference: biasDiff)
    }
}
