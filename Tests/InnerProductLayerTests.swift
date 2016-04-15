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
        let batchSize = 3
        let inputSize = 1024
        let outputSize = 1024

        let input = Matrix<Float>(rows: inputSize, columns: batchSize)
        for i in 0..<inputSize {
            input[i, 0] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            input[i, 1] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            input[i, 2] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let weights = Matrix<Float>(rows: inputSize, columns: outputSize)
        for r in 0..<inputSize {
            for c in 0..<outputSize {
                weights[r, c] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = ValueArray<Float>(count: outputSize)
        for i in 0..<outputSize {
            biases[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let layer = InnerProductLayer(weights: weights, biases: biases, name: "layer")
        try! layer.setupInLibrary(library)

        let queue = device.newCommandQueue()

        let inputBuffer = withPointer(input) { pointer in
            return device.newBufferWithBytes(pointer, length: batchSize * inputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let outputBuffer = device.newBufferWithLength(batchSize * outputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, offset: 0, output: outputBuffer, offset: 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let expectedResult0 = transpose(input)[Interval(integerLiteral: 0), Interval.All] * weights + biases.toRowMatrix()
        let expectedResult1 = transpose(input)[Interval(integerLiteral: 1), Interval.All] * weights + biases.toRowMatrix()
        let expectedResult2 = transpose(input)[Interval(integerLiteral: 2), Interval.All] * weights + biases.toRowMatrix()
        let result = UnsafeMutablePointer<Float>(outputBuffer.contents())
        for i in 0..<outputSize {
            XCTAssertEqualWithAccuracy(result[0 + i * batchSize], expectedResult0[0, i], accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(result[1 + i * batchSize], expectedResult1[0, i], accuracy: 0.0001)
            XCTAssertEqualWithAccuracy(result[2 + i * batchSize], expectedResult2[0, i], accuracy: 0.0001)
        }
    }

}
