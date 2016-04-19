// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Accelerate
import BrainCore
import Upsurge

class InnerProductLayerTests: MetalTestCase {
    func testForward() {
        let batchSize = 1
        let inputSize = 1024
        let outputSize = 1024

        let input = Matrix<Float>(rows: inputSize, columns: batchSize)
        for i in 0..<inputSize {
            input[i, 0] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let weights = Matrix<Float>(rows: inputSize, columns: outputSize)
        for r in 0..<inputSize {
            for c in 0..<outputSize {
                weights[r, c] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = ValueArray<Float>(count: outputSize)
        for i in 0..<outputSize {
            biases[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let layer = InnerProductLayer(weights: weights, biases: biases, name: "layer")
        let sinkLayer = Sink(name: "output", inputSize: outputSize, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        var result = [Float]()
        evaluator.evaluate() { snapshot in
            result = [Float](snapshot.outputOfLayer(layer)!)
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            let expectedResult0 = transpose(input)[Interval(integerLiteral: 0), Interval.All] * weights + biases.toRowMatrix()
            for i in 0..<outputSize {
                XCTAssertEqualWithAccuracy(result[0 + i * batchSize], expectedResult0[0, i], accuracy: 0.0001)
            }
        }
    }

}
