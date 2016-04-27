// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal
import Upsurge

/// Long short-term memory unit (LSTM) recurrent network cell.
public class LSTMNodeLayer: TrainableLayer, BackwardLayer {
    struct Parameters {
        let batchSize: UInt16
        let unitCount: UInt16
        let inputSize: UInt16
        let clipTo: Float
    }

    public let id = NSUUID()
    public let name: String?

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>
    public let clipTo: Float

    public let unitCount: Int
    public let inputSize: Int

    public var outputSize: Int {
        return unitCount
    }

    public var stateSize: Int {
        return 2 * unitCount
    }

    var time: Int

    var previousNode: LSTMNodeLayer?
    var nextNode: LSTMNodeLayer?

    var weightsBuffer: Buffer?
    var weightsDeltasBuffer: Buffer?
    var biasesBuffer: Buffer?
    var biasesDeltasBuffer: Buffer?

    var stateBuffer: Buffer?
    var stateDeltasBuffer: Buffer?

    var activationBuffer: Buffer?
    var activationDeltasBuffer: Buffer?

    var forwardInvocation: Invocation?

    public var forwardInvocations: [Invocation] {
        guard let invocation = forwardInvocation else {
            fatalError("initializeForward needs to be called first")
        }
        return [invocation]
    }

    var backwardActivationsInvocation: Invocation?
    var backwardWeightsInvocation: Invocation?
    var backwardInputsInvocation: Invocation?

    public var backwardInvocations: [Invocation] {
        guard let backwardActivationsInvocation = backwardActivationsInvocation, backwardWeightsInvocation = backwardWeightsInvocation, backwardInputsInvocation = backwardInputsInvocation else {
            fatalError("initializeBackward needs to be called first")
        }
        return [backwardActivationsInvocation, backwardWeightsInvocation, backwardInputsInvocation]
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, time: Int, previousNode: LSTMNodeLayer? = nil, nextNode: LSTMNodeLayer? = nil, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        self.clipTo = clipTo ?? 0
        self.unitCount = biases.count / 4
        self.inputSize = weights.rows - unitCount
        self.time = time
        self.previousNode = previousNode
        self.nextNode = nextNode
        precondition(weights.columns == 4 * unitCount)
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)

        if let previousWeightBuffer = previousNode?.weightsBuffer, previousBiasBuffer = previousNode?.biasesBuffer {
            weightsBuffer = previousWeightBuffer
            biasesBuffer = previousBiasBuffer
        } else if let nextWeightBuffer = nextNode?.weightsBuffer, nextBiasBuffer = nextNode?.biasesBuffer {
            weightsBuffer = nextWeightBuffer
            biasesBuffer = nextBiasBuffer
        } else {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
            biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        }

        stateBuffer = builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))
        activationBuffer = builder.createBuffer(name: "activation", size: batchSize * 4 * unitCount * sizeof(Float))

        let previousStateBuffer: Buffer
        if let stateBuffer = previousNode?.stateBuffer {
            previousStateBuffer = stateBuffer
        } else {
            previousStateBuffer = builder.createBuffer(name: "empty state", size: batchSize * stateSize * sizeof(Float))
        }

        let buffers = [
            builder.inputBuffer,
            weightsBuffer!,
            biasesBuffer!,
            builder.outputBuffer,
            activationBuffer!,
            previousStateBuffer,
            stateBuffer!,
        ]
        forwardInvocation = try builder.createInvocation(
            functionName: "lstm_forward_temporal",
            buffers: buffers,
            values: [UInt(time), params],
            width: unitCount,
            height: batchSize
        )
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        guard let weightsBuffer = weightsBuffer, stateBuffer = stateBuffer, activationBuffer = activationBuffer else {
            preconditionFailure("initializeForward must be called BEFORE initializeBackward.")
        }

        let nextActivationBuffer = nextNode?.activationBuffer ?? builder.createBuffer(name: "activation", size: batchSize * 4 * unitCount * sizeof(Float))
        let nextActivationDeltasBuffer = nextNode?.activationDeltasBuffer ?? builder.createBuffer(name: "activation deltas", size: batchSize * 4 * unitCount * sizeof(Float))
        let nextStateDeltasBuffer = nextNode?.stateDeltasBuffer ?? builder.createBuffer(name: "state deltas", size: batchSize * stateSize * sizeof(Float))
        let previousStateBuffer = previousNode?.stateBuffer ?? builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))

        if let nextNode = nextNode {
            nextNode.activationBuffer = nextActivationBuffer
            nextNode.activationDeltasBuffer = nextActivationDeltasBuffer
            nextNode.stateDeltasBuffer = nextStateDeltasBuffer
        }
        if let previousNode = previousNode {
            previousNode.stateBuffer = previousStateBuffer
        }

        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)

        weightsDeltasBuffer = builder.createBuffer(name: "weights deltas", size: 4 * unitCount * (inputSize + unitCount) * sizeof(Float))
        biasesDeltasBuffer = builder.createBuffer(name: "biases deltas", size: 4 * unitCount * sizeof(Float))
        if stateDeltasBuffer == nil {
            stateDeltasBuffer = builder.createBuffer(name: "state deltas", size: batchSize * stateSize * sizeof(Float))
        }
        if activationDeltasBuffer == nil {
            activationDeltasBuffer = builder.createBuffer(name: "activation deltas", size: batchSize * 4 * unitCount * sizeof(Float))
        }

        let activationsBuffers = [
            builder.outputDeltasBuffer,
            weightsBuffer,
            activationBuffer,
            nextActivationBuffer,
            activationDeltasBuffer!,
            nextActivationDeltasBuffer,
            stateBuffer,
            stateDeltasBuffer!,
            previousStateBuffer,
            nextStateDeltasBuffer,
            ]
        backwardActivationsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_activations",
            buffers: activationsBuffers,
            values: [UInt(time), params],
            width: unitCount,
            height: batchSize
        )

        let weightsBuffers = [
            builder.inputBuffer,
            stateBuffer,
            weightsDeltasBuffer!,
            biasesDeltasBuffer!,
            activationDeltasBuffer!,
            nextActivationDeltasBuffer,
            ]
        backwardWeightsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_weights",
            buffers: weightsBuffers,
            values: [UInt(time), params],
            width: unitCount
        )

        let inputsBuffers = [
            builder.inputDeltasBuffer,
            weightsBuffer,
            activationDeltasBuffer!,
            ]
        backwardInputsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_inputs",
            buffers: inputsBuffers,
            values: [UInt(time), params],
            width: inputSize,
            height: batchSize
        )
    }

    public func encodeParametersUpdate(encodeAction: (values: Buffer, deltas: Buffer) -> Void) {
        guard let weightsDeltasBuffer = weightsDeltasBuffer, weightsBuffer = weightsBuffer else {
            fatalError("LSTM weights were not initialized")
        }
        guard let biasesDeltasBuffer = biasesDeltasBuffer, biasesBuffer = biasesBuffer else {
            fatalError("LSTM biases were not initialized")
        }

        encodeAction(values: weightsBuffer, deltas: weightsDeltasBuffer)
        encodeAction(values: biasesBuffer, deltas: biasesDeltasBuffer)
    }
}
