//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Foundation
import Surge

public class ReLULayer : ForwardLayer {
    public var negativeSlope = 0.0
    public let outputSize: Int

    public init(size: Int) {
        self.outputSize = size
    }
    
    public func forward(input: Blob, inout output: Blob) {
        assert(input.count == outputSize)
        assert(output.count == outputSize)
        for i in 0..<input.count {
            output[i] = max(input[i], 0.0) + negativeSlope * min(input[i], 0.0)
        }
    }

//    public func backward(inputDiff: Blob, output: Blob, inout outputDiff: Blob) {
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
