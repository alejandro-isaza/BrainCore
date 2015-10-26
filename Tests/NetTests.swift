// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import BrainCore
import Upsurge

class NetTests: XCTestCase {
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
        let net = Net()

        let source = Source(data: [1, 1])
        let ip = InnerProductLayer(inputSize: 2, outputSize: 1)
        ip.weights = RealMatrix(rows: 2, columns: 1, elements: [2, 4])
        ip.biases = RealMatrix([[1]])
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(ReLULayer(size: 1), name: "ReLU")
        net.addLayer(sink, name: "sink")

        net.connectLayer("source", toLayer: "inner product")
        net.connectLayer("inner product", toLayer: "ReLU")
        net.connectLayer("ReLU", toLayer: "sink")
        net.forward()

        XCTAssertEqual(sink.data[0], 7)
    }

    func testTwoInputOneOutputNoActivation() {
        let net = Net()

        let source = Source(data: [1, 1])
        let ip = InnerProductLayer(inputSize: 2, outputSize: 1)
        ip.weights = RealMatrix(rows: 2, columns: 1, elements: [2, -4])
        ip.biases = RealMatrix([[1]])
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(ReLULayer(size: 1), name: "ReLU")
        net.addLayer(sink, name: "sink")

        net.connectLayer("source", toLayer: "inner product")
        net.connectLayer("inner product", toLayer: "ReLU")
        net.connectLayer("ReLU", toLayer: "sink")
        net.forward()

        XCTAssertEqual(sink.data[0], 0)
    }
    
    func testInnerProductBackwards() {
        let outputDiff = RealMatrix([[0.1, 0.9, 0.2]])
        let input = RealMatrix([[0.1, 0.7, 0.2]])
        var inputDiff = RealMatrix(rows: 1, columns: 3)
        
        let weights = RealMatrix([
            [1, 5, 6],
            [2, 3, 5],
            [9, 2, 1]
        ])
        let biases = RealArray([-5, 9, 2]).toColumnMatrix()
        
        let ip = InnerProductLayer(weights: weights, biases: biases)
        ip.backward(outputDiff, input: input, inputDiff: &inputDiff)
        ip.update{ (inout param: RealMatrix, inout paramDiff: RealMatrix) in
            param += paramDiff
        }
        
        let expectedWeights = RealMatrix([
            [1.01, 5.07, 6.02],
            [2.09, 3.63, 5.18],
            [9.02, 2.14, 1.04]
        ])
        let expectedBiases = RealMatrix([[-4.9, 9.9, 2.2]])
        let expectedInputDiff = RealMatrix([[3.7, 3.6, 5.3]])

        XCTAssertEqual(expectedWeights, ip.weights)
        XCTAssertEqual(expectedBiases, ip.biases)
        XCTAssertEqual(expectedInputDiff, inputDiff)
    }
    
    func testReluBackwards() {
        var inputDiff = RealMatrix(rows: 1, columns: 3)
        let input = RealMatrix([[5, 3, 9]])
        let outputDiff = RealMatrix([[0.1, -0.2, 0.7]])
        
        let relu = ReLULayer(size: 3)
        relu.backward(outputDiff, input: input, inputDiff: &inputDiff)
        
        XCTAssertEqual(inputDiff, outputDiff)
    }

}
