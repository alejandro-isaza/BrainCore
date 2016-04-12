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

    var resetFunction: MTLComputePipelineState!

    /// Maximum number of instances to enqueue to the GPU at a time
    let instanceCount = 1
    var inflightSemaphore: dispatch_semaphore_t
    var queue: dispatch_queue_t

    var forwardInstance: Instance
    var backwardInstance: Instance
    
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
        forwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)
        backwardInstance = Instance(buffers: net.buffers, device: device, batchSize: batchSize)

        try! setupInLibrary()
    }

    public func setupInLibrary() throws {
        let resetBuffer = library.newFunctionWithName("reset_buffer")!
        resetFunction = try library.device.newComputePipelineStateWithFunction(resetBuffer)

    }

    /// Perform a forward-backward pass on the network. Always call this method from the same serial queue. It may block if there is another run executing.
    ///
    /// - parameter completion: Invoked when the run finishes. It gets passed a snapshot of the network results.
    func run(completion: ((Snapshot) -> Void)) {
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
}
