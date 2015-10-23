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

        let sourceRef = net.addLayer(source)
        let ipRef = net.addLayer(ip)
        let reluRef = net.addLayer(ReLULayer(size: 1))
        let sinkRef = net.addLayer(sink)

        net.connectLayer(sourceRef, toLayer: ipRef)
        net.connectLayer(ipRef, toLayer: reluRef)
        net.connectLayer(reluRef, toLayer: sinkRef)
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

        let sourceRef = net.addLayer(source)
        let ipRef = net.addLayer(ip)
        let reluRef = net.addLayer(ReLULayer(size: 1))
        let sinkRef = net.addLayer(sink)

        net.connectLayer(sourceRef, toLayer: ipRef)
        net.connectLayer(ipRef, toLayer: reluRef)
        net.connectLayer(reluRef, toLayer: sinkRef)
        net.forward()

        XCTAssertEqual(sink.data[0], 0)
    }

}
