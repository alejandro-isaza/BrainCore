// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Accelerate
import BrainCore
import Upsurge

class SGDSolverTests: MetalTestCase {
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
            print(input[0])
            data = input
        }
    }

    func testTrainInnerProduct() {
        let batchSize = 4
        let inputSize = 2
        let outputSize = 1

        let net = Net()

        func generateData() -> (Matrix<Float>, Matrix<Float>) {
            let inputs = Matrix<Float>(rows: batchSize, columns: inputSize)
            let labels = Matrix<Float>(rows: batchSize, columns: outputSize)
            for i in 0..<batchSize {
                for j in 0..<inputSize {
                    inputs[i, j] = round(Float(arc4random()) / Float(UINT32_MAX))
                }
                labels[i, 0] = Float(Int(inputs[i, 0]) ^ Int(inputs[i, 1]))
            }
            return (inputs, labels)
        }

        let (data, label) = generateData()
        let datums = Source(data: data.elements)
        let labels = Source(data: label.elements)

        let weights = Matrix<Float>(rows: inputSize, columns: outputSize)
        for i in 0..<inputSize {
            for j in 0..<outputSize {
                weights[i, j] = 2.0 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }
        let biases = ValueArray<Float>(count: outputSize)

        let ip = InnerProductLayer(weights: weights, biases: biases)
        let loss = L2LossLayer(size: 2 * outputSize)
        let sink = Sink()

        let inputBuffer = net.addBufferWithName("input", size: batchSize * inputSize)
        let ipBuffer = net.addBufferWithName("IP", size: batchSize * 2 * outputSize)
        let outputBuffer = net.addBufferWithName("output", size: 1)

        let sourceLayer = net.addLayer(datums, name: "source")
        let labelLayer = net.addLayer(labels, name: "labels")
        let ipLayer = net.addLayer(ip, name: "inner product")
        let lossLayer = net.addLayer(loss, name: "loss")
        let sinkLayer = net.addLayer(sink, name: "sink")

        net.connectLayer(sourceLayer, toBuffer: inputBuffer)
        net.connectBuffer(inputBuffer, toLayer: ipLayer)
        net.connectLayer(ipLayer, toBuffer: ipBuffer)
        net.connectLayer(labelLayer, toBuffer: ipBuffer, atOffset: outputSize)
        net.connectBuffer(ipBuffer, toLayer: lossLayer)
        net.connectLayer(lossLayer, toBuffer: outputBuffer)
        net.connectBuffer(outputBuffer, toLayer: sinkLayer)

        let solver = SGDSolver(device: device, net: net, batchSize: batchSize, endStep: 30)
        solver.train()
    }
}
