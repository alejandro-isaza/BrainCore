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
    public var biases: RealArray {
        willSet {
            assert(newValue.count == outputSize)
        }
    }

    public init(inputSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        weights = RealMatrix(rows: inputSize, columns: outputSize)
        biases = RealArray(count: outputSize)
    }

    public init(weights: RealMatrix, biases: RealArray) {
        assert(biases.count == weights.columns)
        self.inputSize = weights.rows
        self.outputSize = weights.columns
        self.weights = weights
        self.biases = biases
    }

    public func forward(input: Blob, inout output: Blob) {
        assert(input.count == inputSize)
        assert(output.count == outputSize)
        mul(input, weights, result: &output)
        print(output, biases)
        add(output, biases, result: &output)
    }

//    public func backward(inputDiff: Blob, output: Blob, inout outputDiff: Blob) {
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
