// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import BrainCore
import Upsurge

class NetTests: MetalTestCase {
    class Source : DataLayer {
        var data: Blob
        init(data: Blob) {
            self.data = data
        }
    }

    class Sink : SinkLayer {
        var data: Blob = []
        func consume(input: Blob) {
            data = input
        }
    }

    func testTwoInputOneOutputActivation() {
        let net = try! Net(device: device)

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = Array<Float>([1])

        let ip = try! InnerProductLayer(net: net, weights: weights, biases: biases)
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(try! ReLULayer(net: net, size: 1), name: "ReLU")
        net.addLayer(sink, name: "sink")

        net.connectLayer("source", toLayer: "inner product")
        net.connectLayer("inner product", toLayer: "ReLU")
        net.connectLayer("ReLU", toLayer: "sink")

        let expecation = expectationWithDescription("Net forward pass")
        net.forward(completion: {
            expecation.fulfill()
        })
        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 7)
        }
    }

    func testTwoInputOneOutputNoActivation() {
        let device = self.device
        let net = try! Net(device: device)

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = Array<Float>([1])

        let ip = try! InnerProductLayer(net: net, weights: weights, biases: biases)
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(try! ReLULayer(net: net, size: 1), name: "ReLU")
        net.addLayer(sink, name: "sink")

        net.connectLayer("source", toLayer: "inner product")
        net.connectLayer("inner product", toLayer: "ReLU")
        net.connectLayer("ReLU", toLayer: "sink")

        let expecation = expectationWithDescription("Net forward pass")
        net.forward(completion: {
            expecation.fulfill()
        })
        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 0)
        }
    }

}
