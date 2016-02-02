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

    var device: MTLDevice!
    var library: MTLLibrary!

    override func setUp() {
        let bundle = NSBundle(forClass: self.dynamicType)
        let path = bundle.pathForResource("default", ofType: "metallib")!
        device = MTLCreateSystemDefaultDevice()!
        library = try! device.newLibraryWithFile(path)
    }

    func testTwoInputOneOutputActivation() {
        let net = Net(device: device, library: library)

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = Array<Float>([1])

        let ip = try! InnerProductLayer(library: library, weights: weights, biases: biases)
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(try! ReLULayer(library: library, size: 1), name: "ReLU")
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
        let net = Net(device: device, library: library)

        let source = Source(data: [1, 1])
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, -4])
        let biases = Array<Float>([1])

        let ip = try! InnerProductLayer(library: library, weights: weights, biases: biases)
        let sink = Sink()

        net.addLayer(source, name: "source")
        net.addLayer(ip, name: "inner product")
        net.addLayer(try! ReLULayer(library: library, size: 1), name: "ReLU")
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
