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

        let input = Matrix<Float>(rows: inputSize, columns: batchSize)
        for i in 0..<inputSize {
            for j in 0..<batchSize {
                input[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let label = Matrix<Float>(rows: labelSize, columns: batchSize)
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                label[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let lossLayer = L2LossLayer(size: labelSize)
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
        for i in 0..<inputSize {
            for j in 0..<batchSize {
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

        let input = Matrix<Float>(rows: inputSize, columns: batchSize)
        for i in 0..<inputSize {
            for j in 0..<batchSize {
                input[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }

        let label = Matrix<Float>(rows: labelSize, columns: batchSize)
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                label[i, j] = Float(arc4random()) / Float(UINT32_MAX)
            }
        }
        
        let lossLayer = L2LossLayer(size: labelSize)
        try! lossLayer.setupInLibrary(library)

        let queue = device.newCommandQueue()

        let inputBuffer = device.newBufferWithLength(batchSize * (2 * labelSize) * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let deltasBuffer = device.newBufferWithLength(batchSize * (2 * labelSize) * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = device.newBufferWithLength(batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)

        fillBuffer(inputBuffer, start: 0, withElements: input.elements)
        fillBuffer(inputBuffer, start: input.count, withElements: label.elements)

        var commandBuffer = queue.commandBuffer()
        lossLayer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, offset: 0, output: outputBuffer, offset: 0)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        commandBuffer = queue.commandBuffer()
        lossLayer.encodeBackwardLossInBuffer(commandBuffer, batchSize: batchSize, input: inputBuffer, deltas: deltasBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()


        let expectedResult = Matrix<Float>(rows: 2*labelSize, columns: batchSize)
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                let alpha: Float = 1 / Float(batchSize);
                let diff = input[i, j] - label[i, j];
                expectedResult[i, j] = alpha * diff;
                expectedResult[i+labelSize, j] = alpha * -diff;
            }
        }

        let result = (0..<batchSize*(2*labelSize)).map{ UnsafeMutablePointer<Float>(deltasBuffer.contents())[$0] }
        for i in 0..<labelSize {
            for j in 0..<batchSize {
                XCTAssertEqualWithAccuracy(result[j + i * batchSize], expectedResult[i, j], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(result[j + i * batchSize + batchSize * labelSize], expectedResult[labelSize+i, j], accuracy: 0.0001)
            }
        }
    }

}
