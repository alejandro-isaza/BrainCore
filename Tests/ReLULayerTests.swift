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

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
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

        let dataLayer = Source(name: "input", data: input, batchSize: batchSize)
        let layer = ReLULayer(size: dataSize, name: "ReLU")
        let lossLayer = L2LossLayer(size: dataSize)
        let net = Net.build {
            dataLayer => layer => lossLayer
        }

        let expecation = expectationWithDescription("Net backward pass")
        let trainer = try! Trainer(net: net, device: device, batchSize: batchSize)
        var inputDeltas = [Float]()
        var outputDeltas = [Float]()
        trainer.run() { snapshot in
            inputDeltas = [Float](snapshot.inputDeltasOfLayer(layer)!)
            outputDeltas = [Float](snapshot.outputDeltasOfLayer(layer)!)
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
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

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
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
}
