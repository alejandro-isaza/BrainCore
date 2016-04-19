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

    public let name: String?
    public let id = NSUUID()

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>

    public var inputSize: Int {
        return weights.rows
    }

    public var outputSize: Int {
        return weights.columns
    }

    var weightsBuffer: Buffer?
    var weightDeltasBuffer: Buffer?
    var biasesBuffer: Buffer?
    var biasDeltasBuffer: Buffer?

    var forwardInvocation: Invocation?
    var backwardParameterUpdateInvocation: Invocation?
    var backwardInputUpdateInvocation: Invocation?

    public var forwardInvocations: [Invocation] {
        return [forwardInvocation!]
    }

    public var backwardInvocations: [Invocation] {
        return [
            backwardParameterUpdateInvocation!,
            backwardInputUpdateInvocation!
        ]
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, name: String? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        precondition(biases.count == outputSize)
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        if weightsBuffer == nil {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        }
        if biasesBuffer == nil {
            biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        }

        let buffers = [
            builder.inputBuffer,
            builder.outputBuffer,
            weightsBuffer!,
            biasesBuffer!
        ]

        let params = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        forwardInvocation = try builder.createInvocation(functionName: "inner_product_forward", buffers: buffers, values: [params], width: outputSize, height: batchSize)
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), inputSize: UInt16(inputSize), outputSize: UInt16(outputSize))
        if weightsBuffer == nil {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        }
        if weightDeltasBuffer == nil {
            weightDeltasBuffer = builder.createBuffer(name: "weightDeltas", size: weights.count * sizeof(Float))
        }
        if biasDeltasBuffer == nil {
            biasDeltasBuffer = builder.createBuffer(name: "biasDeltas", size: biases.count * sizeof(Float))
        }

        let paramUpdateBuffers = [
            builder.outputDeltasBuffer,
            builder.inputBuffer,
            weightDeltasBuffer!,
            biasDeltasBuffer!
        ]
        backwardParameterUpdateInvocation = try builder.createInvocation(functionName: "inner_product_backward_params", buffers: paramUpdateBuffers, values: [params], width: outputSize)

        let inputUpdateBuffers = [
            builder.outputDeltasBuffer,
            builder.inputDeltasBuffer,
            weightsBuffer!,
        ]
        backwardInputUpdateInvocation = try builder.createInvocation(functionName: "inner_product_backward_input", buffers: inputUpdateBuffers, values: [params], width: inputSize, height: batchSize)
    }

    public func encodeParametersUpdate(encodeAction: (values: Buffer, deltas: Buffer) -> Void) {
        guard let weightDeltasBuffer = weightDeltasBuffer else {
            fatalError("Inner Product weights were not initialized")
        }
        guard let biasDeltasBuffer = biasDeltasBuffer else {
            fatalError("Inner Product biases were not initialized")
        }

        encodeAction(values: weightsBuffer!, deltas: weightDeltasBuffer)
        encodeAction(values: biasesBuffer!, deltas: biasDeltasBuffer)
    }
}
