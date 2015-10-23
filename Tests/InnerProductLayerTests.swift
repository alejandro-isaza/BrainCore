//  Copyright © 2015 Venture Media Labs. All rights reserved.

import XCTest
import BrainCore
import Upsurge

class InnerProductLayerTests: XCTestCase {

    func testBackwards() {
        let outputDiff = RealMatrix([[0.1, 0.9, 0.2]])
        let input = RealMatrix([[0.1, 0.7, 0.2]])
        var inputDiff = RealMatrix(rows: 1, columns: 3)

        let weights = RealMatrix([
            [1, 5, 6],
            [2, 3, 5],
            [9, 2, 1]
            ])
        let biases = RealArray([-5, 9, 2]).toColumnMatrix()

        let ip = InnerProductLayer(weights: weights, biases: biases)
        ip.backward(outputDiff, input: input, inputDiff: &inputDiff)
        ip.update{ (inout param: RealMatrix, inout paramDiff: RealMatrix) in
            param += paramDiff
        }

        let expectedWeights = RealMatrix([
            [1.01, 5.07, 6.02],
            [2.09, 3.63, 5.18],
            [9.02, 2.14, 1.04]
            ])
        let expectedBiases = RealMatrix([[-4.9, 9.9, 2.2]])
        let expectedInputDiff = RealMatrix([[3.7, 3.6, 5.3]])

        XCTAssertEqual(expectedWeights, ip.weights)
        XCTAssertEqual(expectedBiases, ip.biases)
        XCTAssertEqual(expectedInputDiff, inputDiff)
    }

}
