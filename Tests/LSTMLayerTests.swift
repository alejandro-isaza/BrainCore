// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Accelerate
@testable import BrainCore
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
        let layer = LSTMLayer(weights: weights, biases: biases, name: "layer")
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
        let lstm = LSTMLayer(weights: weights, biases: biases, name: "lstm")
        let layer = RNNLayer(cell: lstm, sequenceLength: 1, name: "lstm")
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

    func testBackward() {
        let device = self.device
        let inputSize = 2
        let unitCount = 1
        let batchSize = 1
        let sequenceSize = 2

        let input = Matrix<Float>(rows: sequenceSize, columns: inputSize, elements: [1, 2, 0.5, 3])
        let labels = Matrix<Float>(rows: sequenceSize, columns: 1, elements: [0.50, 1.25])
        let weights = Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount, elements: [0.95, 0.45, 0.70, 0.60, 0.80, 0.25, 0.45, 0.40, 0.80, 0.15, 0.10, 0.25])
        let biases = ValueArray<Float>([0.65, 0.20, 0.15, 0.10])

        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let labelLayer = Source(name: "input", data: labels.elements, batchSize: batchSize)
        let lstm = LSTMLayer(weights: weights, biases: biases, name: "lstm")
        let layer = RNNLayer(cell: lstm, sequenceLength: sequenceSize, name: "layer")
        let lossLayer = L2LossLayer(size: sequenceSize * unitCount, name: "loss")
        let sinkLayer = Sink(name: "output", inputSize: 1, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer
            [layer, labelLayer] => lossLayer => sinkLayer
        }

        let expecation = expectationWithDescription("Net forward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        trainer.run() { snapshot in
            let output = [Float](snapshot.outputOfLayer(layer)!)
            let state0 = [Float](arrayFromBuffer(layer.cells[0].stateBuffer!.metalBuffer!))
            let state1 = [Float](arrayFromBuffer(layer.cells[1].stateBuffer!.metalBuffer!))
            let activations0 = [Float](arrayFromBuffer(layer.cells[0].activationBuffer!.metalBuffer!))
            let activations1 = [Float](arrayFromBuffer(layer.cells[1].activationBuffer!.metalBuffer!))

            let expectedState0: Float = 0.78572
            let expectedOutput0: Float = 0.53631
            let expectedActivations0: [Float] = [3.20, 1.15, 1.75, 1.5]

            let expectedState1: Float = 1.5176
            let expectedOutput1: Float = 0.77197
            let expectedActivations1: [Float] = [3.9541, 1.2554, 1.9036, 1.7341]

            XCTAssertEqualWithAccuracy(output[0], expectedOutput0, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(output[1], expectedOutput1, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(state0[0], expectedState0, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(state1[0], expectedState1, accuracy: 0.0001)
            for i in 0..<4 {
                XCTAssertEqualWithAccuracy(activations0[i], expectedActivations0[i], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(activations1[i], expectedActivations1[i], accuracy: 0.0001)
            }

            let outputDiff = [Float](snapshot.outputDeltasOfLayer(layer)!)
            let inputDiff = [Float](snapshot.inputDeltasOfLayer(layer)!)
            let stateDiff0 = [Float](arrayFromBuffer(layer.cells[0].stateDeltasBuffer!.metalBuffer!))
            let stateDiff1 = [Float](arrayFromBuffer(layer.cells[1].stateDeltasBuffer!.metalBuffer!))
            let activationsDiff0 = [Float](arrayFromBuffer(layer.cells[0].activationDeltasBuffer!.metalBuffer!))
            let activationsDiff1 = [Float](arrayFromBuffer(layer.cells[1].activationDeltasBuffer!.metalBuffer!))


            let expectedOutputDiff: [Float] = [0.03631, -0.4780]
            let expectedInputDiff: [Float] = [-0.00817, -0.00487, -0.04743, -0.03073]
            let expectedActivationsDiff0: [Float] = [-0.00165, -0.01703, 0, 0.00176]
            let expectedActivationsDiff1: [Float] = [-0.00112, -0.01938, -0.00631, -0.05538]
            let expectedPreviousOutputDiff0: Float = -0.00343
            let expectedPreviousOutputDiff1: Float = -0.01828
            let expectedStateDiff0: Float = -0.05349
            let expectedStateDiff1: Float = -0.07111

            for i in 0..<2 {
                XCTAssertEqualWithAccuracy(outputDiff[i], expectedOutputDiff[i], accuracy: 0.0001)
            }
            for i in 0..<4 {
                XCTAssertEqualWithAccuracy(inputDiff[i], expectedInputDiff[i], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(activationsDiff0[i], expectedActivationsDiff0[i], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(activationsDiff1[i], expectedActivationsDiff1[i], accuracy: 0.0001)
            }
            XCTAssertEqualWithAccuracy(stateDiff0[0], expectedStateDiff0, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(stateDiff1[0], expectedStateDiff1, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(stateDiff0[1], expectedPreviousOutputDiff0, accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(stateDiff1[1], expectedPreviousOutputDiff1, accuracy: 0.0001)


            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { _ in }
    }

}
