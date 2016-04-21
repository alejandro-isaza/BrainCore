// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class L2LossLayer: LossLayer {
    struct Parameters {
        let batchSize: UInt32
        let inputSize: UInt32
    }

    public let size: Int
    public let name: String?
    public let id = NSUUID()

    public var outputSize: Int {
        return 1
    }
    public var inputSize: Int {
        return 2 * size
    }

    var forwardInvocation: Invocation?
    var backwardInvocation: Invocation?

    public var forwardInvocations: [Invocation] {
        guard let forwardInvocation = forwardInvocation else {
            fatalError("initializeForward needs to be called first")
        }
        return [forwardInvocation]
    }

    public var backwardInvocations: [Invocation] {
        guard let backwardInvocation = backwardInvocation else {
            fatalError("initializeBackward needs to be called first")
        }
        return [backwardInvocation]
    }
    
    public init(size: Int, name: String? = nil) {
        self.name = name
        self.size = size
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let buffers = [
            builder.inputBuffer,
            builder.outputBuffer
        ]

        let params = Parameters(batchSize: UInt32(batchSize), inputSize: UInt32(size))
        forwardInvocation = try builder.createInvocation(
            functionName: "l2_loss_forward",
            buffers: buffers,
            values: [params],
            width: batchSize)
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        let params = Parameters(batchSize: UInt32(batchSize), inputSize: UInt32(size))

        let buffers = [
            builder.inputBuffer,
            builder.inputDeltasBuffer
        ]
        backwardInvocation = try builder.createInvocation(
            functionName: "l2_loss_backward",
            buffers: buffers,
            values: [params],
            width: size,
            height: batchSize
        )
    }
}
