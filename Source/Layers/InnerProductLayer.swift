//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Accelerate
import Foundation
import Upsurge

public class InnerProductLayer : ForwardLayer {
    public let inputSize: Int
    public let outputSize: Int
    public var weights: RealMatrix {
        didSet {
            assert(weights.columns == outputSize)
            assert(weights.rows == inputSize)
        }
    }
    private var weightDiff: RealMatrix
    public var biases: RealMatrix {
        willSet {
            assert(newValue.rows == outputSize)
        }
    }
    private var biasDiff: RealMatrix

    public init(inputSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        weights = RealMatrix(rows: inputSize, columns: outputSize)
        biases = RealMatrix(rows: outputSize, columns: 1, repeatedValue: 0.0)
        weightDiff = RealMatrix(rows: inputSize, columns: outputSize)
        biasDiff = RealMatrix(rows: outputSize, columns: 1, repeatedValue: 0.0)
    }

    public init(weights: RealMatrix, biases: RealMatrix) {
        assert(biases.rows == weights.columns)
        self.inputSize = weights.rows
        self.outputSize = weights.columns
        self.weights = weights
        weightDiff = RealMatrix(rows: inputSize, columns: outputSize, repeatedValue: 0.0)
        self.biases = biases
        biasDiff = RealMatrix(rows: outputSize, columns: 1, repeatedValue: 0.0)
    }

    public func forward(input: Blob, inout output: Blob) {
        assert(input.count == inputSize)
        assert(output.count == outputSize)
        mul(input, weights, result: &output)
        add(output, biases.elements, result: &output)
    }

    public func backward(outputDiff: RealMatrix, input: RealMatrix, inout inputDiff: RealMatrix) {
        // Gradient with respect to weight
        weightDiff += outputDiff′ * input

        // Gradient with respect to bias
        biasDiff += outputDiff′

        // Gradient with respect to bottom data
        inputDiff = outputDiff * weights
    }

    public func update(solverUpdateFunction: (inout parameter: RealMatrix, inout parameterDiff: RealMatrix) -> ()) {
        solverUpdateFunction(parameter: &weights, parameterDiff: &weightDiff)
        solverUpdateFunction(parameter: &biases, parameterDiff: &biasDiff)
    }
}
