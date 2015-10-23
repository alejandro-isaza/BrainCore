//  Copyright © 2015 Venture Media Labs. All rights reserved.

import XCTest
import BrainCore
import Upsurge

class ReLULayerTests: XCTestCase {

    func testBackwards() {
        var inputDiff = RealMatrix(rows: 1, columns: 3)
        let input = RealMatrix([[5, 3, 9]])
        let outputDiff = RealMatrix([[0.1, -0.2, 0.7]])

        let relu = ReLULayer(size: 3)
        relu.backward(outputDiff, input: input, inputDiff: &inputDiff)

        XCTAssertEqual(inputDiff, outputDiff)
    }

}
