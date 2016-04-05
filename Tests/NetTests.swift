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
        var batchSize: Int

        var outputSize: Int {
            return data.count / batchSize
        }

        init(data: Blob, batchSize: Int) {
            self.data = data
            self.batchSize = batchSize
        }
    }

    class Sink: SinkLayer {
        var data: Blob = []
        var inputSize: Int
        var batchSize: Int

        init(inputSize: Int, batchSize: Int) {
            self.inputSize = inputSize
            self.batchSize = batchSize
        }

        func consume(input: Blob) {
            data = input
        }
    }

    func testTwoInputOneOutputActivation() {
        let net = Net()

        let source = Source(data: [1, 1, 2, 2], batchSize: 2)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink(inputSize: 1, batchSize: 2)

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
        let runner = try! Runner(net: net, device: device, batchSize: 2)
        runner.forwardPassAction = { buffers in
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

        let source = Source(data: [1, 1], batchSize: 1)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let sink = Sink(inputSize: 1, batchSize: 2)

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
        let runner = try! Runner(net: net, device: device)
        runner.forwardPassAction = { buffers in
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
