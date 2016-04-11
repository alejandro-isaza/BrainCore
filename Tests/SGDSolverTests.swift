// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Accelerate
@testable import BrainCore
import Upsurge

class SGDSolverTests: MetalTestCase {
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

    func testSimpleTrain() {
        let inputSize = 2
        let ip1OutputSize = 2
        let outputSize = 2
        let labelSize = 2
        let batchSize = 1

        let ip1Weights = Matrix<Float>(rows: inputSize, columns: ip1OutputSize, elements: [0.15, 0.25, 0.20, 0.30])
        let ip1Biases: ValueArray<Float> = [0.35, 0.35]

        let ip2Weights = Matrix<Float>(rows: ip1OutputSize, columns: outputSize, elements: [0.40, 0.50, 0.45, 0.55])
        let ip2Biases: ValueArray<Float> = [0.60, 0.60]

        let inputs: ValueArray<Float> = [0.05, 0.10]
        let labels: ValueArray<Float> = [0.01, 0.99]

        let net = Net()

        let data = Source(data: inputs, batchSize: batchSize)
        let label = Source(data: labels, batchSize: batchSize)
        let ip1 = InnerProductLayer(weights: ip1Weights, biases: ip1Biases)
        let sigmoid1 = SigmoidLayer(size: outputSize)
        let ip2 = InnerProductLayer(weights: ip2Weights, biases: ip2Biases)
        let sigmoid2 = SigmoidLayer(size: outputSize)
        let loss = L2LossLayer(size: labelSize)
        let sink = Sink(inputSize: 1, batchSize: batchSize)

        let dataLayer = net.addLayer(data, name: "data")
        let labelLayer = net.addLayer(label, name: "label")
        let ip1Layer = net.addLayer(ip1, name: "ip 1")
        let sigmoid1Layer = net.addLayer(sigmoid1, name: "sigmoid 1")
        let ip2Layer = net.addLayer(ip2, name: "ip 2")
        let sigmoid2Layer = net.addLayer(sigmoid2, name: "sigmoid 2")
        let lossLayer = net.addLayer(loss, name: "loss")
        let sinkLayer = net.addLayer(sink, name: "sink")

        let ip1Buffer = net.addBufferWithName("ip 1")
        let sigmoid1Buffer = net.addBufferWithName("sigmoid 1")
        let ip2Buffer = net.addBufferWithName("ip 2")
        let sigmoid2Buffer = net.addBufferWithName("sigmoid 2")
        let lossBuffer = net.addBufferWithName("loss")
        let sinkBuffer = net.addBufferWithName("sink")

        net.connectLayer(dataLayer, toBuffer: ip1Buffer)
        net.connectBuffer(ip1Buffer, toLayer: ip1Layer)
        net.connectLayer(ip1Layer, toBuffer: sigmoid1Buffer)
        net.connectBuffer(sigmoid1Buffer, toLayer: sigmoid1Layer)
        net.connectLayer(sigmoid1Layer, toBuffer: ip2Buffer)
        net.connectBuffer(ip2Buffer, toLayer: ip2Layer)
        net.connectLayer(ip2Layer, toBuffer: sigmoid2Buffer)
        net.connectBuffer(sigmoid2Buffer, toLayer: sigmoid2Layer)
        net.connectLayer(sigmoid2Layer, toBuffer: lossBuffer, atOffset: 0)
        net.connectLayer(labelLayer, toBuffer: lossBuffer, atOffset: sigmoid2.outputSize)
        net.connectBuffer(lossBuffer, toLayer: lossLayer)
        net.connectLayer(lossLayer, toBuffer: sinkBuffer)
        net.connectBuffer(sinkBuffer, toLayer: sinkLayer)

        let solver = try! SGDSolver(device: device, net: net, batchSize: batchSize, stepCount: 1, learningRate: 0.5, momentum: 1.0)
        let expecation = self.expectationWithDescription("Net forward pass 1")
        solver.stepAction = { snapshot in
            let sinkData = [Float](snapshot.forwardContentsOfBuffer(sinkBuffer))
            sink.consume(Blob(sinkData))

            let sigmoid2Result = snapshot.forwardContentsOfBuffer(lossBuffer)
            let sigmoid2Expected: [Float] = [0.75136507, 0.772928465]
            for i in 0..<outputSize {
                XCTAssertEqualWithAccuracy(sigmoid2Result[i], sigmoid2Expected[i], accuracy: 0.0001)
            }

            let lossResult = sink.data[0]
            let lossExpected: Float = 0.298371109
            XCTAssertEqualWithAccuracy(lossResult, lossExpected, accuracy: 0.0001)

            let ip1ExpectedWeights: [Float] = [0.149780716, 0.24975114, 0.19956143, 0.29950229]
            let ip2ExpectedWeights: [Float] = [0.35891648, 0.511301270, 0.408666186, 0.561370121]

            let ip1ActualWeights: [Float] = arrayFromBuffer(ip1.weightsBuffer)
            let ip2ActualWeights = arrayFromBuffer(ip2.weightsBuffer)

            for i in 0..<4 {
                XCTAssertEqualWithAccuracy(ip1ExpectedWeights[i], ip1ActualWeights[i], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(ip2ExpectedWeights[i], ip2ActualWeights[i], accuracy: 0.0001)
            }
            expecation.fulfill()
        }
        solver.train({ })

        self.waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
        }
    }
}
