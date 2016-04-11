// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
@testable import BrainCore
import Upsurge

class TrainerTests: MetalTestCase {
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

        func nextBatch(batchSize: Int) -> Blob {
            return data
        }
    }

    class Sink: SinkLayer {
        var inputSize: Int
        var batchSize: Int

        init(inputSize: Int, batchSize: Int) {
            self.inputSize = inputSize
            self.batchSize = batchSize
        }

        func consume(input: Blob) {
        }
    }

    func testTwoInputOneOutputActivationForwardBackward() {
        let net = Net()

        let source = Source(data: [1, 1, 2, 2], batchSize: 2)
        let labels = Source(data: [1, 2], batchSize: 2)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let loss = L2LossLayer(size: 1)

        let sourceLayer = net.addLayer(source, name: "source")
        let labelsLayer = net.addLayer(labels, name: "labels")
        let ipBuffer = net.addBufferWithName("ip buff")
        let ipLayer = net.addLayer(ip, name: "ip")
        let lossBuffer = net.addBufferWithName("source buff")
        let lossLayer = net.addLayer(loss, name: "loss")
        let sinkBuffer = net.addBufferWithName("sink buff")
        let sinkLayer = net.addLayer(Sink(inputSize: 1, batchSize: 2), name: "sink")

        net.connectLayer(sourceLayer, toBuffer: ipBuffer)
        net.connectBuffer(ipBuffer, toLayer: ipLayer)
        net.connectLayer(ipLayer, toBuffer: lossBuffer)
        net.connectLayer(labelsLayer, toBuffer: lossBuffer, atOffset: ip.outputSize)
        net.connectBuffer(lossBuffer, toLayer: lossLayer)
        net.connectLayer(lossLayer, toBuffer: sinkBuffer)
        net.connectBuffer(sinkBuffer, toLayer: sinkLayer)

        let ipBufferId = net.nodes.reduce(-1) { val, node in
            if node.layer is InnerProductLayer {
                return node.inputBuffer!.id
            }
            return val
        }

        let expecation = expectationWithDescription("Net forward/backward pass")
        var ipInputDiff = [Float]()
        var ipWeightsDiff = [Float]()
        var ipBiasDiff = [Float]()

        let trainer = try! Trainer(net: net, device: device, batchSize: 2)
        trainer.run() {
            ipInputDiff = arrayFromBuffer(trainer.backwardInstance.buffers[ipBufferId])
            ipWeightsDiff = arrayFromBuffer(ip.weightDiff!)
            ipBiasDiff = arrayFromBuffer(ip.biasDiff!)

            expecation.fulfill()
        }

        waitForExpectationsWithTimeout(5) { error in
            if let error = error {
                XCTFail("trainer.run() failed: \(error)")
            }

            XCTAssertEqual(ipInputDiff, [6, 11, 12, 22])
            XCTAssertEqual(ipWeightsDiff, [14, 14])
            XCTAssertEqual(ipBiasDiff, [8.5])
        }
    }
}
