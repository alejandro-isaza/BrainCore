// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import BrainCore
import Metal
import Upsurge


func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}

class SigmoidLayerTests: MetalTestCase {

    func testForward() {
        let device = self.device
        let dataSize = 512 * 512
        let batchSize = 1

        let data = Blob(count: dataSize)
        for i in 0..<dataSize {
            data[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let dataLayer = Source(name: "input", data: data, batchSize: batchSize)
        let layer = SigmoidLayer(size: dataSize, name: "layer")
        let sinkLayer = Sink(name: "output", inputSize: dataSize, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expecation = expectation(description: "Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expecation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            let result = sinkLayer.data
            for i in 0..<dataSize {
                XCTAssertEqualWithAccuracy(result[i], sigmoid(data[i]), accuracy: 0.001)
            }
        }
    }

    func testBackward() {
        let device = self.device
        let dataSize = 1024 * 1024
        let batchSize = 1

        let input = Blob(count: dataSize)
        for i in 0..<dataSize {
            input[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let label = Blob(count: dataSize)
        for i in 0..<dataSize {
            label[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let dataLayer = Source(name: "input", data: input, batchSize: batchSize)
        let labelLayer = Source(name: "label", data: label, batchSize: batchSize)
        let layer = SigmoidLayer(size: dataSize, name: "layer")
        let lossLayer = L2LossLayer(size: dataSize)
        let sinkLayer = Sink(name: "sink", inputSize: 1, batchSize: 1)
        let net = Net.build {
            dataLayer => layer
            [layer, labelLayer] => lossLayer => sinkLayer
        }

        let expecation = expectation(description: "Net backward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        var inputDeltas = [Float]()
        var outputDeltas = [Float]()
        trainer.run() { snapshot in
            inputDeltas = [Float](snapshot.inputDeltasOfLayer(layer)!)
            outputDeltas = [Float](snapshot.outputDeltasOfLayer(layer)!)
            expecation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            for i in 0..<dataSize {
                XCTAssertEqualWithAccuracy(inputDeltas[i], outputDeltas[i] * sigmoid(input[i]) * (1 - sigmoid(input[i])), accuracy: 0.001)
            }
        }
    }

    func testForwardLargeBatchSize() {
        let device = self.device
        let dataSize = 2
        let batchSize = 4

        let data = Matrix<Float>(rows: batchSize, columns: dataSize)
        for i in 0..<dataSize {
            for j in 0..<batchSize {
                data[j, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let dataLayer = Source(name: "input", data: data.elements, batchSize: batchSize)
        let layer = SigmoidLayer(size: dataSize, name: "layer")
        let sinkLayer = Sink(name: "output", inputSize: dataSize, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expecation = expectation(description: "Net forward pass")

        var result = [Float]()
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        trainer.run() { snapshot in
            result = [Float](snapshot.outputOfLayer(layer)!)
            expecation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            for i in 0..<dataSize {
                for j in 0..<batchSize {
                    XCTAssertEqualWithAccuracy(result[j + i * batchSize], sigmoid(data[j, i]), accuracy: 0.001)
                }
            }
        }
    }
    
}
