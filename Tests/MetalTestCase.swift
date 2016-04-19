// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import BrainCore
import Metal
import XCTest

class MetalTestCase: XCTestCase {
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

        var data: Blob = []

        init(name: String, inputSize: Int, batchSize: Int) {
            self.name = name
            self.inputSize = inputSize
            self.batchSize = batchSize
        }

        func consume(input: Blob) {
            self.data = input
        }
    }
    
    var device: MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            XCTFail("Failed to create a Metal device")
            fatalError("Failed to create a Metal device")
        }

        return d
    }

    var library: MTLLibrary {
        guard let path = NSBundle(forClass: self.dynamicType).pathForResource("default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        return try! device.newLibraryWithFile(path)
    }
}
