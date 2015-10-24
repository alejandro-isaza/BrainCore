//  Copyright © 2015 Venture Media Labs. All rights reserved.

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

}
