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

class LSTMLayerTests: MetalTestCase {
    func testForward() {
        let device = self.device
        let inputSize = 1
        let unitCount = 1
        let batchSize = 1

        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            input[0, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let weights = Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount)
        for r in 0..<inputSize + unitCount {
            for c in 0..<unitCount {
                weights[r, c + 0*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 1*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 2*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 3*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = ValueArray<Float>(count: 4 * unitCount, repeatedValue: 0.0)

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let layer = LSTMLayer(weights: weights, biases: biases, batchSize: 1, name: "layer")
        let sinkLayer = Sink(name: "output", inputSize: unitCount, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            let result = sinkLayer.data
            XCTAssertEqual(result.count, unitCount)

            let inputValue = input[0, 0]
            let expectedActivation = sigmoid(weights[0, 0] * inputValue) * tanh(weights[0, 1] * inputValue)
            let expectedOutput = sigmoid(weights[0, 3] * inputValue) * tanh(expectedActivation)
            XCTAssertEqualWithAccuracy(result[0], expectedOutput, accuracy: 0.001)
        }
    }

    func testForwardLarge() {
        let device = self.device
        let batchSize = 64
        let inputSize = 64
        let unitCount = 128

        let input = Matrix<Float>(rows: batchSize, columns: inputSize)
        for i in 0..<input.rows {
            for j in 0..<input.columns {
                input[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let weights = Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount)
        for r in 0..<inputSize + unitCount {
            for c in 0..<unitCount {
                weights[r, c + 0*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 1*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 2*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 3*unitCount] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = ValueArray<Float>(count: 4 * unitCount, repeatedValue: 0.0)

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let layer = LSTMLayer(weights: weights, biases: biases, batchSize: batchSize, name: "layer")
        let sinkLayer = Sink(name: "output", inputSize: unitCount, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expecation = expectationWithDescription("Net forward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)

        var result = [Float]()
        trainer.run() { snapshot in
            result = [Float](snapshot.outputOfLayer(layer)!)
            XCTAssertEqual(result.count, batchSize * unitCount)
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            for i in 0..<batchSize {
                for j in 0..<unitCount {
                    var inputGate: Float = 0.0
                    var newInput: Float = 0.0
                    var outputGate: Float = 0.0
                    for k in 0..<inputSize {
                        let inputValue = input[i, k]
                        inputGate += weights[k, j] * inputValue
                        newInput += weights[k, j + unitCount] * inputValue
                        outputGate += weights[k, j + 3 * unitCount] * inputValue
                    }
                    let expectedActivation = sigmoid(inputGate) * tanh(newInput)
                    let expectedOutput = sigmoid(outputGate) * tanh(expectedActivation)
                    XCTAssertEqualWithAccuracy(result[i + j * batchSize], expectedOutput, accuracy: 0.001)
                }
            }
        }
    }
}
