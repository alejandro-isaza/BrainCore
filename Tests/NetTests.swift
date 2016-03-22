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
    class Source: DataLayer {
        var data: Blob

        var outputSize: Int {
            return data.count
        }

        init(data: Blob) {
            self.data = data
        }
    }

    class Sink: SinkLayer {
        var data: Blob = []

        func consume(input: Blob) {
            data = input
        }
    }

    func testTwoInputOneOutputActivation() {
        let net = Net()

        let source = Source(data: [1, 1, 2, 2])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink()

        let inputBuffer = net.addBufferWithName("input", size: 4)
        let ipBuffer = net.addBufferWithName("IP", size: 2)
        let outputBuffer = net.addBufferWithName("output", size: 2)

        let sourceLayer = net.addLayer(source, name: "source")
        let ipLayer = net.addLayer(ip, name: "inner product")
        let reluLayer = net.addLayer(ReLULayer(size: 2), name: "ReLU")
        let sinkLayer = net.addLayer(sink, name: "sink")

        net.connectLayer(sourceLayer, toBuffer: inputBuffer)
        net.connectBuffer(inputBuffer, toLayer: ipLayer)
        net.connectLayer(ipLayer, toBuffer: ipBuffer)
        net.connectBuffer(ipBuffer, toLayer: reluLayer)
        net.connectLayer(reluLayer, toBuffer: outputBuffer)
        net.connectBuffer(outputBuffer, toLayer: sinkLayer)

        let expecation = expectationWithDescription("Net forward pass")
        let runner = try! Runner(net: net, device: device, batchSize: 2)
        runner.forwardPassAction = { _ in
            expecation.fulfill()
        }
        runner.forward()

        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 7)
            XCTAssertEqual(sink.data[1], 13)
        }
    }

    func testTwoInputOneOutputNoActivation() {
        let device = self.device
        let net = Net()

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink()

        let inputBuffer = net.addBufferWithName("input", size: 2)
        let ipBuffer = net.addBufferWithName("IP", size: 1)
        let outputBuffer = net.addBufferWithName("output", size: 1)

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
        let runner = try! Runner(net: net, device: device)
        runner.forwardPassAction = { _ in
            expecation.fulfill()
        }
        runner.forward()
        
        waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
            XCTAssertEqual(sink.data[0], 0)
        }
    }

}
