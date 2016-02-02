// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import BrainCore
import Metal
import Upsurge

class ReLULayerTests: XCTestCase {

    func testForward() {
        let dataSize = 1024 * 1024
        let device = MTLCreateSystemDefaultDevice()!

        let bundle = NSBundle(forClass: ReLULayerTests.self)
        let path = bundle.pathForResource("default", ofType: "metallib")!
        let library = try! device.newLibraryWithFile(path)
        let layer = try! ReLULayer(library: library, size: dataSize)

        var data = [Float](count: dataSize, repeatedValue: 0.0)
        for i in 0..<dataSize {
            data[i] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }
        let buffer = data.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: dataSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }

        let queue = device.newCommandQueue()

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, input: buffer, output: buffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        let result = UnsafeBufferPointer<Float>(start: UnsafeMutablePointer(buffer.contents()), count: dataSize)
        for i in 0..<dataSize {
            if data[i] >= 0 {
                XCTAssertEqualWithAccuracy(result[i], data[i], accuracy: 0.001)
            } else {
                XCTAssertEqual(result[i], 0.0)
            }
        }
    }

    func testBackward() {
        let dataSize = 1024 * 1024
        let device = MTLCreateSystemDefaultDevice()!

        let bundle = NSBundle(forClass: ReLULayerTests.self)
        let path = bundle.pathForResource("default", ofType: "metallib")!
        let library = try! device.newLibraryWithFile(path)
        let layer = try! ReLULayer(library: library, size: dataSize)

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
            layer.encodeBackwardInBuffer(commandBuffer, outputDiff: outputDiffBuffer, input: inputBuffer, inputDiff: inputDiffBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = UnsafeBufferPointer<Float>(start: UnsafeMutablePointer(inputDiffBuffer.contents()), count: dataSize)
        for i in 0..<dataSize {
            if input[i] >= 0 {
                XCTAssertEqualWithAccuracy(result[i], outputDiff[i], accuracy: 0.001)
            } else {
                XCTAssertEqual(result[i], 0.0)
            }
        }
    }
}
