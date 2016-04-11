// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
@testable import BrainCore
import Upsurge

class EvaluatorTests: MetalTestCase {
    class Source: DataLayer {
        var data: Blob

        var outputSize: Int {
            return data.count
        }

        init(data: Blob) {
            self.data = data
        }

        func nextBatch(batchSize: Int) -> Blob {
            return data
        }
    }

    class Sink: SinkLayer {
        var data: Blob = []
        var inputSize: Int

        init(inputSize: Int) {
            self.inputSize = inputSize
        }

        func consume(input: Blob) {
            data = input
        }
    }

    func testSplitAndJoin() {
        let data = Matrix<Float>(rows: 1, columns: 4, elements: [1, 1, 2, 2])
        let source1 = Source(data: [1, 1])
        let source2 = Source(data: [2, 2])

        let weights = Matrix<Float>(rows: 4, columns: 10, initializer: { 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0 })
        let biases = ValueArray<Float>(count: 10, repeatedValue: 1.0)
        let ip = InnerProductLayer(weights: weights, biases: biases)

        let sink1 = Sink(inputSize: 6)
        let sink2 = Sink(inputSize: 4)

        let net = [source1, source2] => ip => [(sink1, 0), (sink2, 6)]

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { _ in
            expecation.fulfill()
        }

        let expected = data * weights + biases.toRowMatrix()
        print(expected)
        waitForExpectationsWithTimeout(5) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            for i in 0..<sink1.inputSize {
                XCTAssertEqualWithAccuracy(sink1.data[i], expected.elements[i], accuracy: 0.0001)
            }
            for i in 0..<sink2.inputSize {
                XCTAssertEqualWithAccuracy(sink2.data[i], expected.elements[sink1.inputSize + i], accuracy: 0.0001)
            }
        }
    }

    func testTwoInputOneOutputActivation() {
        let source = Source(data: [1, 2])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink(inputSize: 1)

        let net = source => ip => sink

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { _ in
            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 11.0)
        }
    }

    func testTwoInputOneOutputNoActivation() {
        let device = self.device
        let net = Net()

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink(inputSize: 1)

        let inputBuffer = net.addBufferWithName("input")
        let ipBuffer = net.addBufferWithName("IP")
        let outputBuffer = net.addBufferWithName("output")

        let sourceLayer = net.addLayer(source, name: "source")
        let ipLayer = net.addLayer(ip, name: "inner product")
        let reluLayer = net.addLayer(ReLULayer(size: 1), name: "ReLU")
        let sinkLayer = net.addLayer(sink, name: "sink")

        net.connectLayer(sourceLayer, toBuffer: inputBuffer)
        net.connectBuffer(inputBuffer, toLayer: ipLayer)
        net.connectLayer(ipLayer, toBuffer: ipBuffer)
        net.connectBuffer(ipBuffer, toLayer: reluLayer)
        net.connectLayer(reluLayer, toBuffer: outputBuffer)
        net.connectBuffer(outputBuffer, toLayer: sinkLayer)

        let expecation = expectationWithDescription("Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { _ in
            expecation.fulfill()
        }
        
        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 0)
        }
    }

}
