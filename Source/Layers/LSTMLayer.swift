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
    var reset0Invocation: Invocation?
    var reset1Invocation: Invocation?

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

    /// Creates an LSTM layer with a weights matrix and a biases array.
    ///
    /// - parameter weights:   weight matrix with `4*unitCount` columns and `inputSize + unitCount` rows with the weights for input, input activation, forget, and output in that order and with input weights `W` before recurrent weights `U`.
    /// - parameter biases:    the array of biases of size `4*unitCount` with the biases for input, input activation, forget, and output in that order.
    /// - parameter batchSize: the batch size.
    /// - parameter name:      the layer name.
    /// - parameter clipTo:    optional value to clip activations to.
    ///
    /// - seealso: makeWeightsFromComponents
    public init(weights: Matrix<Float>, biases: ValueArray<Float>, batchSize: Int, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases
        self.clipTo = clipTo ?? 0
        self.unitCount = biases.count / 4
        self.inputSize = weights.rows - unitCount
        precondition(weights.columns == 4 * unitCount)
    }

    /// Make an LSTM weight matrix from separate W and U component matrices.
    public static func makeWeightsFromComponents(Wc: Matrix<Float>, Wf: Matrix<Float>, Wi: Matrix<Float>, Wo: Matrix<Float>, Uc: Matrix<Float>, Uf: Matrix<Float>, Ui: Matrix<Float>, Uo: Matrix<Float>) -> Matrix<Float> {
        let unitCount = Uc.rows
        let inputSize = Wc.rows

        let elements = ValueArray<Float>(count: (inputSize + unitCount) * 4 * unitCount)

        for i in 0..<inputSize {
            let start = i * 4 * unitCount
            elements.replaceRange(0 * unitCount + start..<0 * unitCount + start + unitCount, with: Wi.row(i))
            elements.replaceRange(1 * unitCount + start..<1 * unitCount + start + unitCount, with: Wc.row(i))
            elements.replaceRange(2 * unitCount + start..<2 * unitCount + start + unitCount, with: Wf.row(i))
            elements.replaceRange(3 * unitCount + start..<3 * unitCount + start + unitCount, with: Wo.row(i))
        }
        
        for i in 0..<unitCount {
            let start = (inputSize + i) * 4 * unitCount
            elements.replaceRange(0 * unitCount + start..<0 * unitCount + start + unitCount, with: Ui.row(i))
            elements.replaceRange(1 * unitCount + start..<1 * unitCount + start + unitCount, with: Uc.row(i))
            elements.replaceRange(2 * unitCount + start..<2 * unitCount + start + unitCount, with: Uf.row(i))
            elements.replaceRange(3 * unitCount + start..<3 * unitCount + start + unitCount, with: Uo.row(i))
        }

        return Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount, elements: elements)
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

        reset0Invocation = try builder.createInvocation(
            functionName: "reset_buffer",
            buffers: [state0Buffer!],
            values: [],
            width: stateSize * batchSize)
        reset1Invocation = try builder.createInvocation(
            functionName: "reset_buffer",
            buffers: [state1Buffer!],
            values: [],
            width: stateSize * batchSize)
    }

    /// Reset the internal LSTM state
    public var resetInvocations: [Invocation] {
        return [reset0Invocation!, reset1Invocation!]
    }
}
