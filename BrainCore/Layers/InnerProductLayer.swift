//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Accelerate
import Foundation
import Surge

public class InnerProductLayer : ForwardLayer {
    public let inputSize: Int
    public let outputSize: Int
    public var weights: Matrix<Double> {
        didSet {
            assert(weights.columns == outputSize)
            assert(weights.rows == inputSize)
        }
    }
    public var biases: Matrix<Double> {
        willSet {
            assert(newValue.columns == outputSize)
        }
    }

    public init(inputSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        weights = Matrix<Double>(rows: inputSize, columns: outputSize, repeatedValue: 0.0)
        biases = Matrix<Double>(rows: 1, columns: outputSize, repeatedValue: 0.0)
    }

    public func forward(input: Matrix<Double>, inout output: Matrix<Double>) {
        assert(input.rows == 1 && input.columns == inputSize)
        assert(output.rows == 1 && output.columns == outputSize)
        mul(input, weights, result: &output)
    }

//    public func backward(inputDiff: Matrix<Double>, output: Matrix<Double>, inout outputDiff: Matrix<Double>) {
//        // Gradient with respect to weight
//        cblas_dgemm(CblasTrans, CblasNoTrans, outputSize, inputSize, 1, 1.0, inputDiff, output, 1.0, weightDiff)
//
//        // Gradient with respect to bias
//        cblas_dgemv(CblasTrans, 1, outputSize, 1.0, inputDiff, [1], 1.0, biasDiff)
//
//        // Gradient with respect to bottom data
//        cblas_dgemm(CblasNoTrans, CblasNoTrans, 1, inputSize, outputSize, 1.0, inputDiff, weights, 0.0,outputDiff)
//    }
}
