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

func sigmoid(x: Float) -> Float {
    return 1.0 / (1 + exp(-x))
}

class LSTMLayerTests: MetalTestCase {
    func testForward() {
        let inputSize = 1
        let unitCount = 1

        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            input[0, i] = 1//2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
        }

        let weights = Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount)
        for r in 0..<inputSize + unitCount {
            for c in 0..<unitCount {
                weights[r, c + 0*unitCount] = 0//2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 1*unitCount] = 1//2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 2*unitCount] = 0//2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
                weights[r, c + 3*unitCount] = 0//2 * Float(arc4random()) / Float(UINT32_MAX) - 1.0
            }
        }

        let biases = ValueArray<Float>(count: 4 * unitCount, repeatedValue: 0.0)

        let library = metalLibrary
        let device = library.device
        let layer = try! LSTMLayer(library: library, weights: weights, biases: biases)

        let queue = device.newCommandQueue()

        let inputBuffer = device.newBufferWithBytes(input.pointer, length: inputSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
        let outputBuffer = device.newBufferWithLength(unitCount * sizeof(Float), options: .CPUCacheModeDefaultCache)
        measureBlock {
            layer.reset()
            
            let commandBuffer = queue.commandBuffer()
            layer.encodeForwardInBuffer(commandBuffer, input: inputBuffer, output: outputBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let result = arrayFromBuffer(outputBuffer)
        XCTAssertEqual(result.count, unitCount)

        let inputValue = input[0, 0]
        let expectedActivation = sigmoid(weights[0, 0] * inputValue) * tanh(weights[0, 1] * inputValue)
        let expectedOutput = sigmoid(weights[0, 3] * inputValue) * tanh(expectedActivation)
        XCTAssertEqualWithAccuracy(result[0], expectedOutput, accuracy: 0.001)
    }
    
}
