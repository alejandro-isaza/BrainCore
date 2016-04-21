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

class L2LossLayerTests: MetalTestCase {
    func testForward() {
        let batchSize = 64
        let inputSize = 64
        let labelSize = 64

        let input = Matrix<Float>(rows: inputSize, columns: batchSize)
        for i in 0..<inputSize {
            for j in 0..<batchSize {
                input[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let label = Matrix<Float>(rows: labelSize, columns: batchSize)
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                label[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let labelLayer = Source(name: "label", data: label.elements, batchSize: batchSize)
        let lossLayer = L2LossLayer(size: labelSize, name: "lossLayer")
        let sinkLayer = Sink(name: "output", inputSize: 1, batchSize: batchSize)
        let net = Net.build {
            [dataLayer, labelLayer] => lossLayer => sinkLayer
        }

        let expecation = expectationWithDescription("Net forward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)

        var result = [Float]()
        trainer.run() { snapshot in
            result = [Float](snapshot.outputOfLayer(lossLayer)!)
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            var expectedResult: Float = 0.0
            for i in 0..<inputSize {
                for j in 0..<batchSize {
                    let diff = input[i, j] - label[i, j]
                    expectedResult += diff * diff / 2
                }
            }

            XCTAssertEqualWithAccuracy(sum(result), expectedResult, accuracy: 0.01)
        }
    }

    func testBackward() {
        let batchSize = 64
        let inputSize = 64
        let labelSize = 64

        let input = Matrix<Float>(rows: batchSize, columns: inputSize)
        for i in 0..<inputSize {
            for j in 0..<batchSize {
                input[j, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let label = Matrix<Float>(rows: batchSize, columns: labelSize)
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                label[j, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let labelLayer = Source(name: "label", data: label.elements, batchSize: batchSize)
        let lossLayer = L2LossLayer(size: labelSize, name: "lossLayer")
        let sinkLayer = Sink(name: "output", inputSize: 1, batchSize: batchSize)
        let net = Net.build {
            dataLayer => lossLayer => sinkLayer
            labelLayer => lossLayer
        }

        let expecation = expectationWithDescription("Net backward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        var inputDeltas = [Float]()
        trainer.run() { snapshot in
            inputDeltas = [Float](snapshot.inputDeltasOfLayer(lossLayer)!)
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            let expectedResult = Matrix<Float>(rows: 2*labelSize, columns: batchSize)
            for i in 0..<labelSize {
                for j in 0..<batchSize {
                    let alpha: Float = 1 / Float(batchSize);
                    let diff = input[j, i] - label[j, i];
                    expectedResult[i, j] = alpha * diff;
                    expectedResult[i+labelSize, j] = alpha * -diff;
                }
            }

            let result = inputDeltas[0..<batchSize*(2*labelSize)]
            for i in 0..<labelSize {
                for j in 0..<batchSize {
                    XCTAssertEqualWithAccuracy(result[j + i * batchSize], expectedResult[i, j], accuracy: 0.0001)
                    XCTAssertEqualWithAccuracy(result[j + i * batchSize + batchSize * labelSize], expectedResult[labelSize+i, j], accuracy: 0.0001)
                }
            }
        }
    }

}
