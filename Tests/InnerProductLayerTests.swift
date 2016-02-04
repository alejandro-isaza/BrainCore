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

class InnerProductLayerTests: MetalTestCase {
    func testForward() {
        let inputSize = 1024
        let outputSize = 1024

        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            input[0, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let weights = Matrix<Float>(rows: inputSize, columns: outputSize)
        for r in 0..<inputSize {
            for c in 0..<outputSize {
                weights[r, c] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = Matrix<Float>(rows: 1, columns: outputSize)
        for i in 0..<outputSize {
            biases[0, i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let library = metalLibrary
        let device = library.device
        let layer = try! InnerProductLayer(library: library, weights: weights, biases: biases.row(0))

        let queue = device.newCommandQueue()

        let inputBuffer = device.newBufferWithBytes(input.pointer, length: inputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = device.newBufferWithLength(outputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, input: inputBuffer, output: outputBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let expectedResult = input * weights + biases
        let result = UnsafeMutablePointer<Float>(outputBuffer.contents())
        for i in 0..<outputSize {
            XCTAssertEqualWithAccuracy(result[i], expectedResult[0, i], accuracy: 0.0001)
        }
    }

}
