// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

/// A `Runner` that performs backpropagation passes on a network.
///
/// `Trainer` is optimized for running batches of input data.
///
/// - SeeAlso: `Runner`, `Evaluator`
open class Trainer: Runner {
    var forwardInstance: Instance!
    var backwardInstance: Instance!

    /// Maximum number of instances to enqueue to the GPU at a time.
    let instanceCount = 3
    var inflightSemaphore: DispatchSemaphore
    var queue: DispatchQueue
    
    /// Creates a `Trainer` for the given network definition.
    ///
    /// - Parameter net:    network definition.
    /// - Parameter device: Metal device to use when running.
    public init(net: Net, device: MTLDevice, batchSize: Int) throws {
        queue = DispatchQueue(label: "BrainCore.Evaluator", attributes: [])
        inflightSemaphore = DispatchSemaphore(value: instanceCount)

        try super.init(net: net, device: device, batchSize: batchSize, backwards: true)

        forwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)
        backwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)
    }

    /// Perform a forward-backward pass on the network.
    ///
    /// - Important: Always call this method from the same serial queue. It may block if there is another pass executing.
    ///
    /// - parameter completion: closure to execute when the pass finishes. It gets passed a snapshot of the network results.
    open func run(_ completion: @escaping ((Snapshot) -> Void)) {
        _ = inflightSemaphore.wait(timeout: DispatchTime.distantFuture)

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
                fillBuffer(buffer, start: batchSize * n.outputRange.lowerBound, withElements: dataLayer.nextBatch(batchSize))
            }
            forwardInstance.closeNode(n)
            forwardInstance.openOutputsOf(n)
            forwardInstance.finishNode(n)
        }

        precondition(!forwardInstance.openNodes.isEmpty, "Network is empty")
        queue.async {
            self.processForwardNodes(completion)
            self.processBackwardNodes(completion)
        }
    }

    func processForwardNodes(_ completion: @escaping ((Snapshot) -> Void)) {
        while !forwardInstance.openNodes.isEmpty {
            let node = forwardInstance.openNodes.popLast()!
            if forwardInstance.isClosed(node) {
                continue
            }

            guard let forwardLayer = node.layer as? ForwardLayer else {
                continue
            }

            guard let _ = node.inputBuffer, let _ = node.outputBuffer else {
                preconditionFailure("Layer '\(node.layer.name)' is missing a buffer")
            }

            let buffer = commandQueue.makeCommandBuffer()
            for invocation in forwardLayer.forwardInvocations {
                initializeBuffers(invocation.buffers)
                try! Runner.encode(invocation: invocation, commandBuffer: buffer)
            }

            buffer.addCompletedHandler() { commandBuffer in
                self.queue.async {
                    self.forwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion(Snapshot(net: self.net, forwardBuffers: self.forwardInstance.buffers, backwardBuffers: self.backwardInstance.buffers))
                        self.inflightSemaphore.signal()
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

    func processBackwardNodes(_ completion: @escaping ((Snapshot) -> Void)) {
        while !backwardInstance.openNodes.isEmpty {
            let node = backwardInstance.openNodes.popLast()!
            if backwardInstance.isClosed(node) {
                continue
            }

            guard node.layer is BackwardLayer || node.layer is LossLayer else {
                continue
            }

            guard let _ = node.inputBuffer, let _ = node.outputBuffer else {
                preconditionFailure("Layer '\(node.layer.name)' is missing a buffer")
            }

            let buffer = commandQueue.makeCommandBuffer()
            if let backwardLayer = node.layer as? BackwardLayer {
                for invocation in backwardLayer.backwardInvocations {
                    initializeBuffers(invocation.buffers)
                    try! Runner.encode(invocation: invocation, commandBuffer: buffer)
                }
            }

            buffer.addCompletedHandler() { commandBuffer in
                self.queue.async {
                    self.backwardInstance.finishNode(node)
                    if self.forwardInstance.isFinished() && self.backwardInstance.isFinished() {
                        completion(Snapshot(net: self.net, forwardBuffers: self.forwardInstance.buffers, backwardBuffers: self.backwardInstance.buffers))
                        self.inflightSemaphore.signal()
                    }
                }
            }
            buffer.commit()
            
            backwardInstance.closeNode(node)
            backwardInstance.openInputsOf(node)
        }
    }

    func initializeBuffers(_ buffers: [Buffer]) {
        for buffer in buffers {
            guard let netBuffer = buffer.netBuffer else { continue }
            switch netBuffer.type {
            case .forward:
                if let metalBuffer = forwardInstance.buffers[netBuffer.id] {
                    buffer.metalBuffer = metalBuffer
                }

            case .deltas:
                if let metalBuffer = backwardInstance.buffers[netBuffer.id] {
                    buffer.metalBuffer = metalBuffer
                }

            default: break
            }
        }
    }
}
