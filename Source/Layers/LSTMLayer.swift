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
        public var batchSize: UInt16
        public let unitCount: UInt16
        public let inputSize: UInt16
        public let clipTo: Float

        /// Initialize the parameters for an LSTM cell.
        ///
        /// - parameter unitCount: The number of units in the LSTM cell
        /// - parameter inputSize: The dimensionality of the inputs into the LSTM cell
        /// - parameter clipTo: A float value, if provided the cell state is clipped by this value prior to the cell output activation.
        public init(batchSize: Int, unitCount: Int, inputSize: Int, clipTo: Float? = nil) {
            self.batchSize = UInt16(batchSize)
            self.unitCount = UInt16(unitCount)
            self.inputSize = UInt16(inputSize)
            self.clipTo = clipTo ?? 0
        }
    }

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>
    public var parameters: Parameters

    public var forwardState: MTLComputePipelineState!
    
    public var weightsBuffer: MTLBuffer!
    public var biasesBuffer: MTLBuffer!
    public var stateBuffer: MTLBuffer!
    public var state: Matrix<Float>!
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

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, clipTo: Float? = nil) {
        self.weights = weights
        self.biases = biases

        let unitCount = biases.count / 4
        parameters = Parameters(batchSize: 1, unitCount: unitCount, inputSize: weights.rows - unitCount, clipTo: clipTo)

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
        
        state = Matrix<Float>(rows: Int(parameters.batchSize), columns: stateSize, repeatedValue: 0.0)
        withPointer(state) { pointer in
            self.stateBuffer = library.device.newBufferWithBytes(pointer, length: Int(parameters.batchSize) * stateSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
            self.stateBuffer.label = "LSTMState"
        }
    }

    /// Run one step of LSTM.
    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        parameters.batchSize = UInt16(batchSize)
        self.parametersBuffer = buffer.device.newBufferWithBytes(&parameters, length: sizeof(Parameters), options: .CPUCacheModeDefaultCache)
        self.parametersBuffer.label = "LSTMParameters"

        if batchSize > state.rows {
            state = Matrix<Float>(rows: batchSize, columns: stateSize, repeatedValue: 0.0)
            withPointer(state) { pointer in
                self.stateBuffer = buffer.device.newBufferWithBytes(pointer, length: state.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
                self.stateBuffer.label = "LSTMState"
            }
        }
        
        let encoder = buffer.computeCommandEncoder()
        encoder.label = "LSTMForward"
        encoder.setComputePipelineState(forwardState)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 3)
        encoder.setBuffer(stateBuffer, offset: 0, atIndex: 4)
        encoder.setBuffer(parametersBuffer, offset: 0, atIndex: 5)

        let count = Int(parameters.unitCount)
        let threadsPerGroup = MTLSize(width: forwardState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardState.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    /// Reset the internal LSTM state
    public func reset() {
        let pointer = UnsafeMutablePointer<Float>(stateBuffer.contents())
        for i in 0..<state.count {
            pointer[i] = 0.0
        }
    }
}
