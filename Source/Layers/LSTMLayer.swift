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
    public struct Parameters {
        public let unitCount: UInt16
        public let inputSize: UInt16
        public let clipTo: Float

        /// Initialize the parameters for an LSTM cell.
        ///
        /// - parameter unitCount: The number of units in the LSTM cell
        /// - parameter inputSize: The dimensionality of the inputs into the LSTM cell
        /// - parameter clipTo: A float value, if provided the cell state is clipped by this value prior to the cell output activation.
        public init(unitCount: Int, inputSize: Int, clipTo: Float? = nil) {
            self.unitCount = UInt16(unitCount)
            self.inputSize = UInt16(inputSize)
            self.clipTo = clipTo ?? 0
        }
    }

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>
    public let parameters: Parameters

    public var forwardState: MTLComputePipelineState!
    
    public var weightsBuffer: MTLBuffer!
    public var biasesBuffer: MTLBuffer!
    public var state: MTLBuffer!
    public var parametersBuffer: MTLBuffer!

    public var inputSize: Int {
        return Int(parameters.inputSize)
    }

    public var outputSize: Int {
        return Int(parameters.unitCount)
    }

    public var stateSize: Int {
        return 2 * Int(parameters.unitCount)
    }

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, clipTo: Float? = nil) throws {
        self.weights = weights
        self.biases = biases

        let unitCount = biases.count / 4
        parameters = Parameters(unitCount: unitCount, inputSize: weights.rows - unitCount, clipTo: clipTo)

        precondition(weights.rows == inputSize + unitCount)
        precondition(weights.columns == 4 * unitCount)
        precondition(biases.count == 4 * unitCount)
    }

    public func setupInLibrary(library: MTLLibrary) throws {
        let forwardFunction = library.newFunctionWithName("lstm_forward")!
        forwardState = try library.device.newComputePipelineStateWithFunction(forwardFunction)

        withPointer(weights) { pointer in
            weightsBuffer = library.device.newBufferWithBytes(pointer, length: weights.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        weightsBuffer.label = "LSTMWeights"

        withPointer(biases) { pointer in
            biasesBuffer = library.device.newBufferWithBytes(pointer, length: biases.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        biasesBuffer.label = "LSTMBiases"

        let state = ValueArray<Float>(count: stateSize, repeatedValue: 0.0)
        self.state = library.device.newBufferWithBytes(state.pointer, length: stateSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        self.state.label = "LSTMState"

        var params = parameters
        self.parametersBuffer = library.device.newBufferWithBytes(&params, length: sizeof(Parameters), options: .CPUCacheModeDefaultCache)
        self.parametersBuffer.label = "LSTMParameters"
    }

    /// Run one step of LSTM.
    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        let encoder = buffer.computeCommandEncoder()
        encoder.label = "LSTMForward"
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 3)
        encoder.setBuffer(state, offset: 0, atIndex: 4)
        encoder.setBuffer(parametersBuffer, offset: 0, atIndex: 5)

        let count = Int(parameters.unitCount)
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height:1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    /// Reset the internal LSTM state
    public func reset() {
        let pointer = UnsafeMutablePointer<Float>(state.contents())
        for i in 0..<stateSize {
            pointer[i] = 0.0
        }
    }
}
