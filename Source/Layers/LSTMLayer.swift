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
public class LSTMLayer: BackwardLayer {
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

    var stateBuffers: [Buffer]?
    var stateDeltasBuffers: [Buffer]?
    var activationBuffers: [Buffer]?
    var activationDeltasBuffers: [Buffer]?
    var weightsBuffer: Buffer?
    var weightDeltasBuffers: [Buffer]?
    var biasesBuffer: Buffer?
    var biasDeltasBuffers: [Buffer]?


    public let unitCount: Int
    public let inputSize: Int
    public let timeSteps: Int

    public var outputSize: Int {
        return unitCount
    }

    public var stateSize: Int {
        return 2 * unitCount
    }

    var T = 0

    var forwardInvocationsOverTime: [Invocation]?

    public var forwardInvocations: [Invocation] {
        guard let invocation = forwardInvocationsOverTime?[T] else {
            fatalError("initializeForward needs to be called first")
        }
        T = (T + 1) % timeSteps
        return [invocation]
    }

    var backwardActivationsInvocationsOverTime: [Invocation]?
    var backwardWeightsInvocationsOverTime: [Invocation]?
    var backwardInputsInvocationsOverTime: [Invocation]?

    public var backwardInvocations: [Invocation] {
        guard let activationsInvocation = backwardActivationsInvocationsOverTime?[T] else {
            fatalError("initializeBackward needs to be called first")
        }
        guard let weightsInvocation = backwardWeightsInvocationsOverTime?[T] else {
            fatalError("initializeBackward needs to be called first")
        }
        guard let inputsInvocation = backwardInputsInvocationsOverTime?[T] else {
            fatalError("initializeBackward needs to be called first")
        }
        T = (T + 1) % timeSteps
        return [activationsInvocation, weightsInvocation, inputsInvocation]
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, batchSize: Int, timeSteps: Int = 1, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        self.clipTo = clipTo ?? 0
        self.unitCount = biases.count / 4
        self.inputSize = weights.rows - unitCount
        self.timeSteps = timeSteps
        precondition(weights.columns == 4 * unitCount)
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)
        if weightsBuffer == nil {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        }
        if biasesBuffer == nil {
            biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        }
        if stateBuffers == nil {
            stateBuffers = [Buffer]()
            for time in 0..<timeSteps {
                stateBuffers!.append(builder.createBuffer(name: "state\(time)", size: batchSize * stateSize * sizeof(Float)))
            }
        }
        if activationBuffers == nil {
            activationBuffers = [Buffer]()
            for time in 0..<timeSteps {
                activationBuffers!.append(builder.createBuffer(name: "activation\(time)", size: batchSize * unitCount * sizeof(Float)))
            }
        }

        forwardInvocationsOverTime = [Invocation]()

        for timeStep in 0..<timeSteps {
            let previousTimestep = timeStep - 1 >= 0 ? timeStep - 1 : timeSteps - 1

            let buffers = [
                builder.inputBuffer,
                weightsBuffer!,
                biasesBuffer!,
                builder.outputBuffer,
                activationBuffers![timeStep],
                stateBuffers![previousTimestep],
                stateBuffers![timeStep]
            ]
            forwardInvocationsOverTime!.append(try builder.createInvocation(
                functionName: "lstm_forward",
                buffers: buffers,
                values: [params],
                width: unitCount,
                height: batchSize
            ))
        }
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)
        if weightsBuffer == nil {
            weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        }
        if biasesBuffer == nil {
            biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        }
        if stateBuffers == nil {
            stateBuffers = [Buffer]()
            for time in 0..<timeSteps {
                stateBuffers!.append(builder.createBuffer(name: "state\(time)", size: batchSize * stateSize * sizeof(Float)))
            }
        }
        if stateDeltasBuffers == nil {
            stateDeltasBuffers = [Buffer]()
            for time in 0..<timeSteps {
                stateDeltasBuffers!.append(builder.createBuffer(name: "state\(time)", size: batchSize * stateSize * sizeof(Float)))
            }
        }
        if activationBuffers == nil {
            activationBuffers = [Buffer]()
            for time in 0..<timeSteps {
                activationBuffers!.append(builder.createBuffer(name: "activation\(time)", size: batchSize * unitCount * sizeof(Float)))
            }
        }
        if activationDeltasBuffers == nil {
            activationDeltasBuffers = [Buffer]()
            for time in 0..<timeSteps {
                activationDeltasBuffers!.append(builder.createBuffer(name: "activation\(time)", size: batchSize * unitCount * sizeof(Float)))
            }
        }
        if weightDeltasBuffers == nil {
            weightDeltasBuffers = [Buffer]()
            for time in 0..<timeSteps {
                weightDeltasBuffers!.append(builder.createBuffer(name: "weightDeltas\(time)", size: weights.count * sizeof(Float)))
            }
        }
        if biasDeltasBuffers == nil {
            biasDeltasBuffers = [Buffer]()
            for time in 0..<timeSteps {
                biasDeltasBuffers!.append(builder.createBuffer(name: "biasDeltas\(time)", size: biases.count * sizeof(Float)))
            }
        }

        backwardActivationsInvocationsOverTime = [Invocation]()
        backwardWeightsInvocationsOverTime = [Invocation]()
        backwardInputsInvocationsOverTime = [Invocation]()

        for timeStep in (0..<timeSteps).reverse() {
            let previousTimestep = timeStep - 1 >= 0 ? timeStep - 1 : timeSteps - 1
            let nextTimestep = timeStep + 1 < timeSteps ? timeStep + 1 : 0

            var buffers = [
                builder.outputDeltasBuffer,
                weightsBuffer!,
                activationBuffers![timeStep],
                activationBuffers![nextTimestep],
                activationDeltasBuffers![timeStep],
                activationDeltasBuffers![nextTimestep],
                stateBuffers![timeStep],
                stateDeltasBuffers![timeStep],
                stateBuffers![previousTimestep],
                stateDeltasBuffers![nextTimestep],
            ]
            backwardActivationsInvocationsOverTime!.append(try builder.createInvocation(
                functionName: "lstm_backward_activations",
                buffers: buffers,
                values: [params],
                width: unitCount,
                height: batchSize
            ))

            buffers = [
                builder.inputBuffer,
                stateBuffers![timeStep],
                weightDeltasBuffers![timeStep],
                biasDeltasBuffers![timeStep],
                activationBuffers![timeStep],
                activationBuffers![nextTimestep],
            ]
            backwardWeightsInvocationsOverTime!.append(try builder.createInvocation(
                functionName: "lstm_backward_weights",
                buffers: buffers,
                values: [params],
                width: unitCount,
                height: batchSize
            ))

            buffers = [
                builder.inputDeltasBuffer,
                weightsBuffer!,
                activationDeltasBuffers![timeStep],
            ]
            backwardInputsInvocationsOverTime!.append(try builder.createInvocation(
                functionName: "lstm_backward_inputs",
                buffers: buffers,
                values: [params],
                width: inputSize,
                height: batchSize
            ))
        }

    }
}
