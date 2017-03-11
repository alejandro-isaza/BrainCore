// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
@testable import BrainCore
import Metal
import Upsurge

class ReLULayerTests: MetalTestCase {

    func testForward() {
        let device = self.device
        let dataSize = 1024 * 1024
        let batchSize = 1

        let data = ValueArray<Float>(count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            data[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let dataLayer = Source(name: "input", data: data, batchSize: batchSize)
        let layer = ReLULayer(size: dataSize, name: "ReLU")
        let sinkLayer = Sink(name: "output", inputSize: dataSize, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer => sinkLayer
        }

        let expectation = self.expectation(description: "Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            let result = sinkLayer.data
            for i in 0..<dataSize {
                if data[i] >= 0 {
                    XCTAssertEqualWithAccuracy(result[i], data[i], accuracy: 0.001)
                } else {
                    XCTAssertEqual(result[i], 0.0)
                }
            }
        }
    }

    func testBackward() {
        let device = self.device
        let dataSize = 1024 * 1024
        let batchSize = 1

        let input = ValueArray<Float>(count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            input[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let label = ValueArray<Float>(count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            label[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let dataLayer = Source(name: "input", data: input, batchSize: batchSize)
        let labelLayer = Source(name: "label", data: label, batchSize: batchSize)
        let layer = ReLULayer(size: dataSize, name: "ReLU")
        let lossLayer = L2LossLayer(size: dataSize)
        let sinkLayer = Sink(name: "sink", inputSize: lossLayer.outputSize, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer
            [layer, labelLayer] => lossLayer => sinkLayer
        }

        let expectation = self.expectation(description: "Net backward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        var inputDeltas = [Float]()
        var outputDeltas = [Float]()
        trainer.run() { snapshot in
            inputDeltas = [Float](snapshot.inputDeltasOfLayer(layer)!)
            outputDeltas = [Float](snapshot.outputDeltasOfLayer(layer)!)
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            for i in 0..<dataSize {
                if input[i] >= 0 {
                    XCTAssertEqualWithAccuracy(inputDeltas[i], outputDeltas[i], accuracy: 0.001)
                } else {
                    XCTAssertEqual(inputDeltas[i], 0.0)
                }
            }
        }
    }
    
    func testForwardLargeBatchSize() {
        let device = self.device
        let dataSize = 16 * 1024
        let batchSize = 64

        let data = Matrix<Float>(rows: batchSize, columns: dataSize)
        for i in 0..<batchSize {
            for j in 0..<dataSize {
                data[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }
        let label = Matrix<Float>(rows: batchSize, columns: dataSize)
        for i in 0..<batchSize {
            for j in 0..<dataSize {
                label[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let dataLayer = Source(name: "input", data: data.elements, batchSize: batchSize)
        let labelLayer = Source(name: "label", data: label.elements, batchSize: batchSize)
        let layer = ReLULayer(size: dataSize, name: "ReLU")
        let lossLayer = L2LossLayer(size: dataSize)
        let sinkLayer = Sink(name: "output", inputSize: 1, batchSize: batchSize)
        let net = Net.build {
            dataLayer => layer
            [layer, labelLayer] => lossLayer => sinkLayer
        }

        let expectation = self.expectation(description: "Net forward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)

        var result = [Float]()
        trainer.run() { snapshot in
            result = [Float](snapshot.outputOfLayer(layer)!)
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5) { error in
            for i in 0..<dataSize {
                for j in 0..<batchSize {
                    if data[j, i] >= 0 {
                        XCTAssertEqualWithAccuracy(result[j + i * batchSize], data[j, i], accuracy: 0.001)
                    } else {
                        XCTAssertEqual(result[j + i * batchSize], 0.0)
                    }
                }
            }
        }
    }
}
