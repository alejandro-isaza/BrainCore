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
public class LSTMLayer: ForwardLayer {
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

    var forwardInvocation0: Invocation?
    var forwardInvocation1: Invocation?

    var weightsBuffer: Buffer?
    var biasesBuffer: Buffer?
    var state0Buffer: Buffer?
    var state1Buffer: Buffer?

    public var stateBuffer: Buffer? {
        if currentState == 0 {
            return state0Buffer
        } else {
            return state1Buffer
        }
    }

    public var forwardInvocations: [Invocation] {
        guard let forwardInvocation0 = forwardInvocation0, forwardInvocation1 = forwardInvocation1 else {
            fatalError("initializeForward needs to be called first")
        }
        let invocation = currentState == 0 ? forwardInvocation0 : forwardInvocation1
        currentState = (currentState + 1) % 2
        return [invocation]
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, batchSize: Int, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        self.clipTo = clipTo ?? 0
        self.unitCount = biases.count / 4
        self.inputSize = weights.rows - unitCount
        precondition(weights.columns == 4 * unitCount)
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt16(batchSize), unitCount: UInt16(unitCount), inputSize: UInt16(inputSize), clipTo: clipTo)
        weightsBuffer = builder.createBuffer(name: "weights", elements: weights)
        biasesBuffer = builder.createBuffer(name: "biases", elements: biases)
        state0Buffer = builder.createBuffer(name: "state0", size: batchSize * stateSize * sizeof(Float))
        state1Buffer = builder.createBuffer(name: "state1", size: batchSize * stateSize * sizeof(Float))

        let buffers0 = [
            builder.inputBuffer,
            weightsBuffer!,
            biasesBuffer!,
            builder.outputBuffer,
            state0Buffer!,
            state1Buffer!
        ]
        forwardInvocation0 = try builder.createInvocation(
            functionName: "lstm_forward_simple",
            buffers: buffers0,
            values: [params],
            width: unitCount,
            height: batchSize
        )

        let buffers1 = [
            builder.inputBuffer,
            weightsBuffer!,
            biasesBuffer!,
            builder.outputBuffer,
            state1Buffer!,
            state0Buffer!
        ]
        forwardInvocation1 = try builder.createInvocation(
            functionName: "lstm_forward_simple",
            buffers: buffers1,
            values: [params],
            width: unitCount,
            height: batchSize)
    }

    /// Reset the internal LSTM state
    public func reset() {
        let pointer0 = UnsafeMutablePointer<Float>(state0Buffer!.metalBuffer!.contents())
        for i in 0..<state0Buffer!.metalBuffer!.length / sizeof(Float) {
            pointer0[i] = 0.0
        }
        let pointer1 = UnsafeMutablePointer<Float>(state1Buffer!.metalBuffer!.contents())
        for i in 0..<state1Buffer!.metalBuffer!.length / sizeof(Float) {
            pointer1[i] = 0.0
        }
        currentState = 0
    }
}
