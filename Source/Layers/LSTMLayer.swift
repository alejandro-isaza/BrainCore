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
        var batchSize: UInt16
        let unitCount: UInt16
        let inputSize: UInt16
        let clipTo: Float

        /// Initialize the parameters for an LSTM cell.
        ///
        /// - parameter batchSize: The batch size
        /// - parameter unitCount: The number of units in the LSTM cell
        /// - parameter inputSize: The dimensionality of the inputs into the LSTM cell
        /// - parameter clipTo: A float value, if provided the cell state is clipped by this value prior to the cell output activation.
        init(batchSize: Int, unitCount: Int, inputSize: Int, clipTo: Float? = nil) {
            self.batchSize = UInt16(batchSize)
            self.unitCount = UInt16(unitCount)
            self.inputSize = UInt16(inputSize)
            self.clipTo = clipTo ?? 0
        }
    }

    var parameters: Parameters
    public let name: String?
    public let id = NSUUID()

    public let weights: Matrix<Float>
    public let biases: ValueArray<Float>
    
    public var weightsBuffer: MTLBuffer!
    public var biasesBuffer: MTLBuffer!
    public var stateBuffer: MTLBuffer!
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

    public init(weights: Matrix<Float>, biases: ValueArray<Float>, batchSize: Int, name: String? = nil, clipTo: Float? = nil) {
        self.name = name
        self.weights = weights
        self.biases = biases

        let unitCount = biases.count / 4
        parameters = Parameters(batchSize: batchSize, unitCount: unitCount, inputSize: weights.rows - unitCount, clipTo: clipTo)

        precondition(weights.rows == inputSize + unitCount)
        precondition(weights.columns == 4 * unitCount)
        precondition(biases.count == 4 * unitCount)
    }

    static let forwardFunctionName = "lstm_forward"
    public var forwardFunction: MTLComputePipelineState!

    public func setupInLibrary(library: MTLLibrary) throws {
        let function = library.newFunctionWithName(LSTMLayer.forwardFunctionName)!
        forwardFunction = try library.device.newComputePipelineStateWithFunction(function)

        self.weightsBuffer = createBuffer(inDevice: library.device, fromTensor: weights, withLabel: "LSTMWeights")
        self.biasesBuffer = createBuffer(inDevice: library.device, fromTensor: biases, withLabel: "LSTMBiases")

        let state = Matrix<Float>(rows: Int(parameters.batchSize), columns: stateSize, repeatedValue: 0.0)
        self.stateBuffer = createBuffer(inDevice: library.device, fromTensor: state, withLabel: "LSTMState")
    }

    /// Run one step of LSTM.
    public func encodeForwardInBuffer(buffer: MTLCommandBuffer, batchSize: Int, input: MTLBuffer, offset inputOffset: Int, output: MTLBuffer, offset outputOffset: Int) {
        parameters.batchSize = UInt16(batchSize)
        parametersBuffer = createBuffer(inDevice: buffer.device, fromPointer: &parameters, ofSize: sizeof(Parameters), withLabel: "LSTMParameters")

        let encoder = buffer.computeCommandEncoder()
        encoder.label = "LSTMForward"
        encoder.setComputePipelineState(forwardFunction)
        encoder.setBuffer(input, offset: inputOffset * sizeof(Float), atIndex: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, atIndex: 2)
        encoder.setBuffer(output, offset: outputOffset * sizeof(Float), atIndex: 3)
        encoder.setBuffer(stateBuffer, offset: 0, atIndex: 4)
        encoder.setBuffer(parametersBuffer, offset: 0, atIndex: 5)

        let count = Int(parameters.unitCount)
        let threadsPerGroup = MTLSize(width: forwardFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / forwardFunction.threadExecutionWidth + 1, height: batchSize, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    /// Reset the internal LSTM state
    public func reset() {
        let pointer = UnsafeMutablePointer<Float>(stateBuffer!.contents())
        for i in 0..<stateBuffer!.length / sizeof(Float) {
            pointer[i] = 0.0
        }
    }
}
