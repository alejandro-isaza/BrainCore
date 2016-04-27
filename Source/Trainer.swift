// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class Trainer: Runner {
    var forwardInstance: Instance!
    var backwardInstance: Instance!

    /// Maximum number of instances to enqueue to the GPU at a time
    let instanceCount = 3
    var inflightSemaphore: dispatch_semaphore_t
    var queue: dispatch_queue_t
    
    public init(net: Net, device: MTLDevice, batchSize: Int) throws {
        queue = dispatch_queue_create("BrainCore.Evaluator", DISPATCH_QUEUE_SERIAL)
        inflightSemaphore = dispatch_semaphore_create(instanceCount)

        try super.init(net: net, device: device, batchSize: batchSize, backwards: true)

        forwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)
        backwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)
    }

    /// Perform a forward-backward pass on the network. Always call this method from the same serial queue. It may block if there is another run executing.
    ///
    /// - parameter completion: Invoked when the run finishes. It gets passed a snapshot of the network results.
    public func run(completion: ((Snapshot) -> Void)) {
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)

        forwardInstance.reset()
        backwardInstance.reset()

        commandQueue.insertDebugCaptureBoundary()

        // Collect all data
        for n in net.dataNodes.values {
            let dataLayer = n.layer as! DataLayer

            if let netBuffer = n.outputBuffer {
                guard let buffer = forwardInstance.buffers[netBuffer.id] else {
                    fatalError("Output buffer for \(dataLayer.name) not found.")
                }
                fillBuffer(buffer, start: batchSize * n.outputRange.startIndex, withElements: dataLayer.nextBatch(batchSize))
            }
            forwardInstance.closeNode(n)
            forwardInstance.openOutputsOf(n)
            forwardInstance.finishNode(n)
        }

        precondition(!forwardInstance.openNodes.isEmpty, "Network is empty")
        dispatch_async(queue) {
            self.processForwardNodes(completion)
            self.processBackwardNodes(completion)
        }
    }

    func processForwardNodes(completion: ((Snapshot) -> Void)) {
        while !forwardInstance.openNodes.isEmpty {
            let node = forwardInstance.openNodes.popLast()!
            if forwardInstance.isClosed(node) {
                continue
            }

            guard let forwardLayer = node.layer as? ForwardLayer else {
                continue
            }

            guard let _ = node.inputBuffer, _ = node.outputBuffer else {
                preconditionFailure("Layer '\(node.layer.name)' is missing a buffer")
            }

            let buffer = commandQueue.commandBuffer()
            for invocation in forwardLayer.forwardInvocations {
                initializeBuffers(invocation.buffers)
                try! encode(invocation: invocation, forNode: node, commandBuffer: buffer)
            }

            buffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    self.forwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion(Snapshot(net: self.net, forwardBuffers: self.forwardInstance.buffers, backwardBuffers: self.backwardInstance.buffers))
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
    }

    func processBackwardNodes(completion: ((Snapshot) -> Void)) {
        while !backwardInstance.openNodes.isEmpty {
            let node = backwardInstance.openNodes.popLast()!
            if backwardInstance.isClosed(node) {
                continue
            }

            guard node.layer is BackwardLayer || node.layer is LossLayer else {
                continue
            }

            guard let _ = node.inputBuffer, _ = node.outputBuffer else {
                preconditionFailure("Layer '\(node.layer.name)' is missing a buffer")
            }

            let buffer = commandQueue.commandBuffer()
            if let backwardLayer = node.layer as? BackwardLayer {
                for invocation in backwardLayer.backwardInvocations {
                    initializeBuffers(invocation.buffers)
                    try! encode(invocation: invocation, forNode: node, commandBuffer: buffer)
                }
            }

            buffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    self.backwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion(Snapshot(net: self.net, forwardBuffers: self.forwardInstance.buffers, backwardBuffers: self.backwardInstance.buffers))
                        dispatch_semaphore_signal(self.inflightSemaphore)
                    }
                }
            }
            buffer.commit()
            
            backwardInstance.closeNode(node)
            backwardInstance.openInputsOf(node)
        }
    }

    func initializeBuffers(buffers: [Buffer]) {
        for buffer in buffers {
            guard let netBuffer = buffer.netBuffer else { continue }
            switch netBuffer.type {
            case .Forward:
                if let metalBuffer = forwardInstance.buffers[netBuffer.id] {
                    buffer.metalBuffer = metalBuffer
                }

            case .Deltas:
                if let metalBuffer = backwardInstance.buffers[netBuffer.id] {
                    buffer.metalBuffer = metalBuffer
                }

            default: break
            }
        }
    }
}
