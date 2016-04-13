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
        let name: String?
        let id = NSUUID()
        var data: Blob
        var batchSize: Int

        var outputSize: Int {
            return data.count / batchSize
        }

        init(name: String, data: Blob, batchSize: Int) {
            self.name = name
            self.data = data
            self.batchSize = batchSize
        }

        func nextBatch(batchSize: Int) -> Blob {
            return data
        }
    }

    class Sink: SinkLayer {
        let name: String?
        let id = NSUUID()
        var inputSize: Int
        var batchSize: Int

        init(name: String, inputSize: Int, batchSize: Int) {
            self.name = name
            self.inputSize = inputSize
            self.batchSize = batchSize
        }

        func consume(input: Blob) {
        }
    }

    func testTwoInputOneOutputActivationForwardBackward() {
        let source = Source(name: "source", data: [1, 1, 2, 2], batchSize: 2)
        let labels = Source(name: "labels", data: [1, 2], batchSize: 2)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])

        let ip = InnerProductLayer(weights: weights, biases: biases, name: "ip")
        let loss = L2LossLayer(size: 1, name: "loss")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: 2)

        let net = Net.build({
            source => ip
            [ip, labels] => loss => sink
        })

        let expecation = expectationWithDescription("Net forward/backward pass")
        var ipInputDiff = [Float]()
        var ipWeightsDiff = [Float]()
        var ipBiasDiff = [Float]()

        let trainer = try! Trainer(net: net, device: device, batchSize: 2)
        trainer.run() { snapshot in
            ipInputDiff = [Float](snapshot.inputDeltasOfLayer(ip)!)
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
