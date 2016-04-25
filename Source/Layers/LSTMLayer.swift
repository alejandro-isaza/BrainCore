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
public class LSTMLayer: TrainableLayer, BackwardLayer {
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

    var currentState = 0
    var time: Int?
    var continuous: Bool

    var previousLSTM: LSTMLayer?
    var nextLSTM: LSTMLayer?

    var weightsBuffer: Buffer?
    var weightsDeltasBuffer: Buffer?
    var biasesBuffer: Buffer?
    var biasesDeltasBuffer: Buffer?

    var stateBuffer: Buffer?
    var stateDeltasBuffer: Buffer?

    var activationBuffer: Buffer?
    var activationDeltasBuffer: Buffer?

    var forwardInvocation0: Invocation?
    var forwardInvocation1: Invocation?

    public var forwardInvocations: [Invocation] {
        if continuous {
            guard let invocation = currentState == 0 ? forwardInvocation0 : forwardInvocation1 else {
                fatalError("initializeForward needs to be called first")
            }
            return [invocation]
        } else {
            guard let invocation = forwardInvocation0 else {
                fatalError("initializeForward needs to be called first")
            }
            return [invocation]
        }
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

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, continuous: Bool = true, time: Int? = nil, previousLSTM: LSTMLayer? = nil, nextLSTM: LSTMLayer? = nil, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        self.clipTo = clipTo ?? 0
        self.unitCount = biases.count / 4
        self.inputSize = weights.rows - unitCount
        self.time = time
        self.previousLSTM = previousLSTM
        self.nextLSTM = nextLSTM
        self.continuous = continuous
        precondition(weights.columns == 4 * unitCount)
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)
        weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        stateBuffer = builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))
        activationBuffer = builder.createBuffer(name: "activation", size: batchSize * 4 * unitCount * sizeof(Float))

        let previousStateBuffer: Buffer
        if let previousLSTM = previousLSTM {
            previousStateBuffer = previousLSTM.stateBuffer!
        } else {
            previousStateBuffer = builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))
        }

        let buffers0 = [
            builder.inputBuffer,
            weightsBuffer!,
            biasesBuffer!,
            builder.outputBuffer,
            activationBuffer!,
            previousStateBuffer,
            stateBuffer!,
        ]
        forwardInvocation0 = try builder.createInvocation(
            functionName: "lstm_forward",
            buffers: buffers0,
            values: [UInt(time ?? 0), params],
            width: unitCount,
            height: batchSize
        )

        let buffers1 = [
            builder.inputBuffer,
            weightsBuffer!,
            biasesBuffer!,
            builder.outputBuffer,
            activationBuffer!,
            stateBuffer!,
            previousStateBuffer,
        ]
        forwardInvocation1 = try builder.createInvocation(
            functionName: "lstm_forward",
            buffers: buffers1,
            values: [UInt(time ?? 0), params],
            width: unitCount,
            height: batchSize
        )
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        guard previousLSTM == nil || nextLSTM == nil else {
            fatalError("LSTMLayer must be wrapped in an RNNLayer ")
        }

        let nextActivationBuffer = nextLSTM?.activationBuffer ?? builder.createBuffer(name: "activation", size: batchSize * 4 * unitCount * sizeof(Float))
        let nextActivationDeltasBuffer = nextLSTM?.activationDeltasBuffer ?? builder.createBuffer(name: "activation deltas", size: batchSize * 4 * unitCount * sizeof(Float))
        let nextStateDeltasBuffer = nextLSTM?.stateDeltasBuffer ?? builder.createBuffer(name: "state deltas", size: batchSize * stateSize * sizeof(Float))
        let previousStateBuffer = previousLSTM?.stateBuffer ?? builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))

        if let nextLSTM = nextLSTM {
            nextLSTM.activationBuffer = nextActivationBuffer
            nextLSTM.activationDeltasBuffer = nextActivationDeltasBuffer
            nextLSTM.stateDeltasBuffer = nextStateDeltasBuffer
        }
        if let previousLSTM = previousLSTM {
            previousLSTM.stateBuffer = previousStateBuffer
        }

        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)
        if weightsBuffer == nil {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        }
        if biasesBuffer == nil {
            biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        }
        if stateBuffer == nil {
            stateBuffer = builder.createBuffer(name: "state", size: batchSize * stateSize * sizeof(Float))
        }
        if activationBuffer == nil {
            activationBuffer = builder.createBuffer(name: "activation", size: batchSize * 4 * unitCount * sizeof(Float))
        }
        if weightsDeltasBuffer == nil {
            weightsDeltasBuffer = builder.createBuffer(name: "weights deltas", elements: weights)
        }
        if biasesDeltasBuffer == nil {
            biasesDeltasBuffer = builder.createBuffer(name: "biases deltas", elements: biases)
        }
        if stateDeltasBuffer == nil {
            stateDeltasBuffer = builder.createBuffer(name: "state deltas", size: batchSize * stateSize * sizeof(Float))
        }
        if activationDeltasBuffer == nil {
            activationDeltasBuffer = builder.createBuffer(name: "activation deltas", size: batchSize * 4 * unitCount * sizeof(Float))
        }

        let activationsBuffers = [
            builder.outputDeltasBuffer,
            weightsBuffer!,
            activationBuffer!,
            nextActivationBuffer,
            activationDeltasBuffer!,
            nextActivationDeltasBuffer,
            stateBuffer!,
            stateDeltasBuffer!,
            previousStateBuffer,
            nextStateDeltasBuffer,
            ]
        backwardActivationsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_activations",
            buffers: activationsBuffers,
            values: [UInt(time ?? 0), params],
            width: unitCount,
            height: batchSize
        )

        let weightsBuffers = [
            builder.inputBuffer,
            stateBuffer!,
            weightsDeltasBuffer!,
            biasesDeltasBuffer!,
            activationBuffer!,
            nextActivationBuffer,
            ]
        backwardWeightsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_weights",
            buffers: weightsBuffers,
            values: [UInt(time ?? 0), params],
            width: unitCount,
            height: batchSize
        )

        let inputsBuffers = [
            builder.inputDeltasBuffer,
            weightsBuffer!,
            activationDeltasBuffer!,
            ]
        backwardInputsInvocation = try builder.createInvocation(
            functionName: "lstm_backward_inputs",
            buffers: inputsBuffers,
            values: [UInt(time ?? 0), params],
            width: inputSize,
            height: batchSize
        )
    }

    public func encodeParametersUpdate(encodeAction: (values: Buffer, deltas: Buffer) -> Void) {
        guard let weightDeltasBuffer = weightsDeltasBuffer else {
            fatalError("LSTM weights were not initialized")
        }
        guard let biasDeltasBuffer = biasesDeltasBuffer else {
            fatalError("LSTM biases were not initialized")
        }

        encodeAction(values: weightsBuffer!, deltas: weightDeltasBuffer)
        encodeAction(values: biasesBuffer!, deltas: biasDeltasBuffer)
    }
}
