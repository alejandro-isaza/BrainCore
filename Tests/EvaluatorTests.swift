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

    func testSplitAndJoin() {
        let data = Matrix<Float>(rows: 1, columns: 4, elements: [1, 1, 2, 2])
        let source1 = Source(name: "source1", data: [1, 1], batchSize: 1)
        let source2 = Source(name: "source2", data: [2, 2], batchSize: 1)

        let weights = Matrix<Float>(rows: 4, columns: 10, initializer: { 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0 })
        let biases = ValueArray<Float>(count: 10, repeatedValue: 1.0)
        let ip = InnerProductLayer(weights: weights, biases: biases, name: "ip")

        let sink1 = Sink(name: "sink1", inputSize: 6, batchSize: 1)
        let sink2 = Sink(name: "sink2", inputSize: 4, batchSize: 1)

        let net = Net.build({
            [source1, source2] => ip => [sink1, sink2]
        })

        let expectation = self.expectation(description: "Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { snapshot in
            expectation.fulfill()
        }

        let expected = data * weights + biases.toRowMatrix()
        waitForExpectations(timeout: 5) { error in
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
        let source = Source(name: "source", data: [1, 2], batchSize: 1)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases, name: "ip")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: 1)

        let net = Net.build({
            source => ip => sink
        })

        let expectation = self.expectation(description: "Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { _ in
            expectation.fulfill()
        }

        waitForExpectations(timeout: 2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 11.0)
        }
    }

    func testTwoInputOneOutputNoActivation() {
        let source = Source(name: "source", data: [1, 1], batchSize: 1)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases, name: "ip")
        let relu = ReLULayer(size: 1, name: "relu")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: 1)

        let net = Net.build({
            source => ip => relu => sink
        })

        let expectation = self.expectation(description: "Net forward pass")
        let evaluator = try! Evaluator(net: net, device: device)
        evaluator.evaluate() { _ in
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 0)
        }
    }

}
