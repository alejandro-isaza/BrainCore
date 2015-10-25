// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

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
