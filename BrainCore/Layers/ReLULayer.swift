//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Foundation
import Surge

public class ReLULayer : ForwardLayer {
    public var negativeSlope = 0.0

    public func forward(input: Matrix<Double>, inout output: Matrix<Double>) {
        assert(input.rows == 1)
        assert(output.rows == 1)
        assert(input.columns == output.columns)

        let N = input.columns
        for i in 0..<N {
            output[0, i] = max(input[0, i], 0.0) + negativeSlope * min(input[0, i], 0.0)
        }
    }

//    public func backward(inputDiff: Matrix<Double>, output: Matrix<Double>, inout outputDiff: Matrix<Double>) {
//        assert(inputDiff.dimensions.count == 1)
//        assert(output.dimensions.count == 1)
//        assert(outputDiff.dimensions.count == 1)
//        assert(inputDiff.width == output.width && inputDiff.width == output.width)
//
//        let N = inputDiff.width
//        for i in 0..<N {
//            let pos = output[i] > 0 ? 1.0 : 0.0
//            let neg = output[i] <= 0 ? 1.0 : 0.0
//            outputDiff[i] = inputDiff[i] * (pos + negativeSlope * neg)
//        }
//    }
}
