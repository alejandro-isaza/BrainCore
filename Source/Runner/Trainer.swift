// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class Trainer {
    public let batchSize: Int
    public let net: Net

    let device: MTLDevice
    var library: MTLLibrary!
    var commandQueue: MTLCommandQueue

    /// Maximum number of instances to enqueue to the GPU at a time
    let instanceCount = 1
    var inflightSemaphore: dispatch_semaphore_t
    var queue: dispatch_queue_t

    var forwardInstance: TrainerInstance
    var backwardInstance: TrainerInstance
    
    public init(net: Net, device: MTLDevice, batchSize: Int) throws {
        self.batchSize = batchSize
        self.net = net
        self.device = device

        commandQueue = device.newCommandQueue()
        commandQueue.label = "BrainCore.Trainer"

        guard let path = NSBundle(forClass: self.dynamicType).pathForResource("default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        library = try device.newLibraryWithFile(path)

        for node in net.nodes {
            if let forwardLayer = node.layer as? ForwardLayer {
                try forwardLayer.setupInLibrary(library)
            }
        }

        inflightSemaphore = dispatch_semaphore_create(instanceCount)
        queue = dispatch_queue_create("BrainCore.Trainer", DISPATCH_QUEUE_SERIAL)
        forwardInstance = TrainerInstance(buffers: net.buffers, device: device, batchSize: batchSize)
        backwardInstance = TrainerInstance(buffers: net.buffers, device: device, batchSize: batchSize)
    }

    /// Perform a forward/backward pass on the network. Always call this method from the same serial queue. It may block if there is another run executing.
    ///
    /// - parameter completion: Invoked when the run finishes. It gets passed an array of buffers that can be used to inspect intermediate results. These buffers are short-lived, you should make a copy of the contents if you need them.
    func run(completion: (() -> Void)) {
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)

        forwardInstance.reset()
        backwardInstance.reset()

        commandQueue.insertDebugCaptureBoundary()

        // Collect all data
        for n in net.dataNodes {
            let dataLayer = n.layer as! DataLayer

            if let netBuffer = n.outputBuffer {
                let buffer = forwardInstance.buffers[netBuffer.id]
                fillBuffer(buffer, start: batchSize * n.outputOffset, withElements: dataLayer.nextBatch(batchSize))
            }
            forwardInstance.closeNode(n)
            forwardInstance.finishNode(n)
        }

        dispatch_async(queue) {
            self.processNodes(completion)
        }
    }

    func processNodes(completion: (() -> Void)) {
        while !forwardInstance.openNodes.isEmpty {
            let node = forwardInstance.openNodes.popLast()!
            if forwardInstance.isClosed(node) {
                continue
            }

            guard let forwardLayer = node.layer as? ForwardLayer else {
                continue
            }

            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }

            let inputBuffer = forwardInstance.buffers[input.id]
            let outputBuffer = forwardInstance.buffers[output.id]

            let buffer = commandQueue.commandBuffer()
            forwardLayer.encodeForwardInBuffer(buffer,
                                               batchSize: batchSize,
                                               input: inputBuffer,
                                               offset: batchSize * node.inputOffset,
                                               output: outputBuffer,
                                               offset: batchSize * node.outputOffset)

            buffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    self.forwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion()
                        dispatch_semaphore_signal(self.inflightSemaphore)
                    }
                }
            }
            buffer.commit()
            forwardInstance.closeNode(node)
            forwardInstance.openOutputsOf(node)

            if node.layer is LossLayer {
                backwardInstance.openNodes.append(node)
            }
        }

        while !backwardInstance.openNodes.isEmpty {
            let node = backwardInstance.openNodes.popLast()!
            if backwardInstance.isClosed(node) {
                continue
            }

            guard node.layer is BackwardLayer || node.layer is LossLayer else {
                continue
            }

            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }

            let inputBuffer = forwardInstance.buffers[input.id]
            let inputDiffBuffer = backwardInstance.buffers[input.id]

            let buffer = commandQueue.commandBuffer()
            if let backwardLayer = node.layer as? BackwardLayer {
                let outputDiffBuffer = backwardInstance.buffers[output.id]

                backwardLayer.encodeBackwardInBuffer(buffer,
                                                     batchSize: batchSize,
                                                     outputDiff: outputDiffBuffer,
                                                     input: inputBuffer,
                                                     inputDiff: inputDiffBuffer)
            } else if let lossLayer = node.layer as? LossLayer {
                lossLayer.encodeBackwardLossInBuffer(buffer,
                                                     batchSize: batchSize,
                                                     input: inputBuffer,
                                                     deltas: inputDiffBuffer)
            }

            buffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    self.backwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion()
                        dispatch_semaphore_signal(self.inflightSemaphore)
                    }
                }
            }
            buffer.commit()
            
            backwardInstance.closeNode(node)
            backwardInstance.openInputsOf(node)
        }
    }
}
