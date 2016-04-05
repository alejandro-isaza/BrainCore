// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import BrainCore
import Metal
import Upsurge

func sigmoid(x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}

class SigmoidLayerTests: MetalTestCase {

    func testForward() {
        let dataSize = 512 * 512

        let device = self.device
        let layer = SigmoidLayer(size: dataSize)
        try! layer.setupInLibrary(library)

        var data = [Float](count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            data[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let inputbuffer = data.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let outputbuffer = device.newBufferWithLength(dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)

        let queue = device.newCommandQueue()

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, batchSize: 1, input: inputbuffer, offset: 0, output: outputbuffer, offset: 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = UnsafeBufferPointer<Float>(start: UnsafeMutablePointer(outputbuffer.contents()), count: dataSize)
        for i in 0..<dataSize {
            XCTAssertEqualWithAccuracy(result[i], sigmoid(data[i]), accuracy: 0.001)
        }
    }

    func testBackward() {
        let dataSize = 1024 * 1024

        let device = self.device
        let layer = SigmoidLayer(size: dataSize)
        try! layer.setupInLibrary(library)

        var input = [Float](count: dataSize, repeatedValue: 0.0)
        var outputDiff = [Float](count: dataSize, repeatedValue: 0.0)
        let inputDiff = [Float](count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            input[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            outputDiff[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let outputDiffBuffer = outputDiff.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let inputBuffer = input.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let inputDiffBuffer = inputDiff.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }

        let queue = device.newCommandQueue()

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeBackwardInBuffer(commandBuffer, batchSize: 1, outputDiff: outputDiffBuffer, input: inputBuffer, inputDiff: inputDiffBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = arrayFromBuffer(inputDiffBuffer)
        for i in 0..<dataSize {
            XCTAssertEqualWithAccuracy(result[i], outputDiff[i] * sigmoid(input[i]) * (1 - sigmoid(input[i])), accuracy: 0.001)
        }
    }

    func testForwardLargeBatchSize() {
        let batchSize = 64
        let dataSize = 64 * 1024

        let device = self.device
        let layer = SigmoidLayer(size: dataSize)
        try! layer.setupInLibrary(library)

        var data = [Float](count: batchSize * dataSize, repeatedValue: 0.0)
        for i in 0..<batchSize * dataSize {
            data[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let inputbuffer = data.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: batchSize * dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let outputbuffer = device.newBufferWithLength(batchSize * dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)

        let queue = device.newCommandQueue()

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, batchSize: batchSize, input: inputbuffer, offset: 0, output: outputbuffer, offset: 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = UnsafeBufferPointer<Float>(start: UnsafeMutablePointer(outputbuffer.contents()), count: batchSize * dataSize)
        for i in 0..<batchSize * dataSize {
            XCTAssertEqualWithAccuracy(result[i], sigmoid(data[i]), accuracy: 0.001)
        }
    }
    
}
