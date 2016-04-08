// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class BackwardRunnerInstance {
    var diffBuffers: [MTLBuffer]
    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    var batchSize: Int

    private var queue: dispatch_queue_t

    init(diffBuffers: [NetBuffer], device: MTLDevice, batchSize: Int) {
        self.batchSize = batchSize
        self.queue = dispatch_queue_create("BrainCore.ForwardRunnerInstance", DISPATCH_QUEUE_SERIAL)

        self.diffBuffers = [MTLBuffer]()
        self.diffBuffers.reserveCapacity(diffBuffers.count)
        for buffer in diffBuffers {
            let mtlBuffer = device.newBufferWithLength(buffer.size * batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
            mtlBuffer.label = "\(buffer.name)Buffer"
            self.diffBuffers.append(mtlBuffer)
        }
    }

    func processNodes(commandQueue: MTLCommandQueue, forwardInstance: ForwardRunnerInstance, terminateBackwardPass: (BackwardRunnerInstance, ForwardRunnerInstance) -> Void) {
        dispatch_sync(queue) {
            self.processNodesInQueue(commandQueue, forwardInstance: forwardInstance, terminateBackwardPass: terminateBackwardPass)
        }
    }

    private func processNodesInQueue(commandQueue: MTLCommandQueue, forwardInstance: ForwardRunnerInstance, terminateBackwardPass: (BackwardRunnerInstance, ForwardRunnerInstance) -> Void) {
        while !openNodes.isEmpty {
            let node = openNodes.popLast()!
            if closedNodes.contains(node) {
                continue
            }

            guard node.layer is BackwardLayer || node.layer is LossLayer else {
                continue
            }

            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }

            let inputBuffer = forwardInstance.buffers[input.id]
            let inputDiffBuffer = diffBuffers[input.id]

            let buffer = commandQueue.commandBuffer()
            if let backwardLayer = node.layer as? BackwardLayer {
                let outputDiffBuffer = diffBuffers[output.id]
                
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
                    self.finishNode(node)
                    if self.isFinished() {
                        terminateBackwardPass(self, forwardInstance)
                    }
                }
            }
            buffer.commit()

            closeNode(node)
        }
    }

    func reset() {
        openNodes.removeAll(keepCapacity: true)
        closedNodes.removeAll(keepCapacity: true)
        finishedNodes.removeAll(keepCapacity: true)
    }

    func isNodeReady(node: NetNode) -> Bool {
        guard let buffer = node.outputBuffer else {
            return true
        }

        for n in buffer.outputNodes {
            if !closedNodes.contains(n) {
                return false
            }
        }
        return true
    }

    func closeNode(node: NetNode) {
        closedNodes.insert(node)

        if let buffer = node.inputBuffer {
            let newOpenNodes = buffer.inputNodes.filter{ isNodeReady($0) }
            openNodes.appendContentsOf(newOpenNodes)
        }
    }

    func finishNode(node: NetNode) {
        finishedNodes.insert(node)
    }

    func isFinished() -> Bool {
        return openNodes.isEmpty && closedNodes == finishedNodes
    }
}
