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
        let inputSize = 4
        let labelSize = 4

        let input = Matrix<Float>(rows: batchSize, columns: inputSize)
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                input[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let label = Matrix<Float>(rows: batchSize, columns: labelSize)
        for i in 0..<batchSize {
            for j in 0..<labelSize {
                label[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let lossLayer = L2LossLayer(size: inputSize + labelSize)
        try! lossLayer.setupInLibrary(library)

        let queue = device.newCommandQueue()

        let inputBuffer = device.newBufferWithLength(batchSize * (inputSize + labelSize) * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = device.newBufferWithLength(batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)

        fillBuffer(inputBuffer, start: 0, withElements: input.elements)
        fillBuffer(inputBuffer, start: input.count, withElements: label.elements)

        let commandBuffer = queue.commandBuffer()
        lossLayer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, offset: 0, output: outputBuffer, offset: 0)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        var expectedResult: Float = 0.0
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                let diff = input[i, j] - label[i, j]
                expectedResult += diff * diff / 2
            }
        }

        let result = ValueArray<Float>((0..<batchSize).map{ UnsafeMutablePointer<Float>(outputBuffer.contents())[$0] })
        XCTAssertEqualWithAccuracy(sum(result), expectedResult, accuracy: 0.0001)
    }

    func testBackward() {
        let batchSize = 3
        let inputSize = 4
        let labelSize = 4

        let input = Matrix<Float>(rows: batchSize, columns: inputSize)
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                input[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let label = Matrix<Float>(rows: batchSize, columns: labelSize)
        for i in 0..<batchSize {
            for j in 0..<labelSize {
                label[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let lossLayer = L2LossLayer(size: inputSize + labelSize)
        try! lossLayer.setupInLibrary(library)

        let queue = device.newCommandQueue()

        let inputBuffer = device.newBufferWithLength(batchSize * (inputSize + labelSize) * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let inputDiffBuffer = device.newBufferWithLength(batchSize * (inputSize + labelSize) * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = device.newBufferWithLength(batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputDiffBuffer = device.newBufferWithLength(batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)

        fillBuffer(inputBuffer, start: 0, withElements: input.elements)
        fillBuffer(inputBuffer, start: input.count, withElements: label.elements)

        var commandBuffer = queue.commandBuffer()
        lossLayer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, offset: 0, output: outputBuffer, offset: 0)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        commandBuffer = queue.commandBuffer()
        lossLayer.encodeBackwardInBuffer(commandBuffer, batchSize: batchSize, outputDiff: outputDiffBuffer, input: inputBuffer, inputDiff: inputDiffBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()


        let expectedResult = Matrix<Float>(rows: batchSize, columns: inputSize+labelSize)
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                let alpha: Float = 1 / Float(batchSize);
                let diff = input[i, j] - label[i, j];
                expectedResult[i, j] = alpha * diff;
                expectedResult[i, j+inputSize] = alpha * -diff;
            }
        }

        let result = (0..<batchSize*(inputSize+labelSize)).map{ UnsafeMutablePointer<Float>(inputDiffBuffer.contents())[$0] }
        for i in 0..<batchSize {
            for j in 0..<inputSize {
                XCTAssertEqualWithAccuracy(result[i*inputSize + j], expectedResult[i, j], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(result[i*inputSize + j + batchSize*inputSize], expectedResult[i, inputSize+j], accuracy: 0.0001)
            }
        }
    }

}
