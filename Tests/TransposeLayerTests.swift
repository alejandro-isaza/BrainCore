// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
@testable import BrainCore
import Metal
import Upsurge


class TransposeLayerTests: MetalTestCase {

    func testForward() {
        let batchSize = 64
        let dataSize = 64 * 64

        let device = self.device
        let layer = TransposeLayer(size: dataSize, name: "Transpose")
        try! layer.setupInLibrary(library)

        let data = Matrix<Float>(rows: batchSize, columns: dataSize)
        for i in 0..<batchSize {
            for j in 0..<dataSize {
            data[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let inputbuffer = data.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: data.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let outputbuffer = device.newBufferWithLength(data.count * sizeof(Float), options: .CPUCacheModeDefaultCache)

        let queue = device.newCommandQueue()

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputbuffer, offset: 0, output: outputbuffer, offset: 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = UnsafePointer<Float>(outputbuffer.contents())
        let expected = transpose(data)
        for i in 0..<batchSize * dataSize {
            XCTAssertEqualWithAccuracy(result[i], expected.elements[i], accuracy: 0.001)
        }
    }
    
}
