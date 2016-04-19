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
        let device = self.device
        let batchSize = 64
        let dataSize = 64 * 64

        let data = Matrix<Float>(rows: batchSize, columns: dataSize)
        for i in 0..<batchSize {
            for j in 0..<dataSize {
                data[i, j] = 2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let inputMetalBuffer = data.withUnsafeBufferPointer { pointer in
            return device.newBufferWithBytes(pointer.baseAddress, length: data.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        }
        let inputBuffer = Buffer(name: "input", size: data.count * sizeof(Float), metalBuffer: inputMetalBuffer, offset: 0)

        let outputMetalBuffer = device.newBufferWithLength(data.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = Buffer(name: "output", size: data.count * sizeof(Float), metalBuffer: outputMetalBuffer, offset: 0)

        let builder = ForwardInvocationBuilder(device: self.device, library: self.library, inputBuffer: inputBuffer, outputBuffer: outputBuffer)
        let layer = TransposeLayer(size: dataSize, name: "Transpose")
        try! layer.initializeForward(builder: builder, batchSize: batchSize)

        let queue = device.newCommandQueue()
        let invocation = layer.forwardInvocation!

        measureBlock {
            let commandBuffer = queue.commandBuffer()
            let encoder = commandBuffer.computeCommandEncoder()
            encoder.setComputePipelineState(invocation.pipelineState)

            for (index, buffer) in invocation.buffers.enumerate() {
                encoder.setBuffer(buffer.metalBuffer!, offset: buffer.metalBufferOffset * batchSize, atIndex: index)
            }

            for (index, value) in invocation.values.enumerate() {
                var mutableValue = value
                encoder.setBytes(&mutableValue, length: sizeofValue(value), atIndex: index + invocation.buffers.count)
            }

            let threadWidth = invocation.pipelineState.threadExecutionWidth
            let threadsPerGroup = MTLSize(width: threadWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (invocation.width - 1) / threadWidth + 1, height: invocation.height, depth: invocation.depth)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
            
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = UnsafePointer<Float>(outputMetalBuffer.contents())
        let expected = transpose(data)
        for i in 0..<batchSize * dataSize {
            XCTAssertEqualWithAccuracy(result[i], expected.elements[i], accuracy: 0.001)
        }
    }
    
}
