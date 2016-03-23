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

class L2LossLayerTests: MetalTestCase {
    func testForward() {
        let batchSize = 3
        let inputSize = 2
        let labelSize = 1
        let outputSize = 1

        let input = Matrix<Float>(rows: batchSize, columns: inputSize)
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                input[i, j] = round(Float(arc4random()) / Float(UINT32_MAX))
            }
        }

        let lossLayer = L2LossLayer(size: inputSize + labelSize)
        try! lossLayer.setupInLibrary(library)

        let queue = device.newCommandQueue()

        let inputBuffer = withPointer(input) { pointer in
            return device.newBufferWithBytes(pointer, length: batchSize * inputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let outputBuffer = device.newBufferWithLength(batchSize * outputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        measureBlock {
            let commandBuffer = queue.commandBuffer()
            lossLayer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, offset: 0, output: outputBuffer, offset: 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let expectedResult = input[0, 0]
        let result = UnsafeMutablePointer<Float>(outputBuffer.contents())
        for i in 0..<outputSize {
            XCTAssertEqualWithAccuracy(result[i], expectedResult, accuracy: 0.0001)
        }
    }
    
}
