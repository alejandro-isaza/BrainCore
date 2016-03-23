// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class ForwardRunnerInstance {
    var buffers: [MTLBuffer]
    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    var batchSize: Int

    var queue: dispatch_queue_t

    init(buffers: [NetBuffer], device: MTLDevice, batchSize: Int) {
        self.batchSize = batchSize
        self.queue = dispatch_queue_create("BrainCore.ForwardRunnerInstance", DISPATCH_QUEUE_SERIAL)

        self.buffers = [MTLBuffer]()
        self.buffers.reserveCapacity(buffers.count)
        for buffer in buffers {
            let mtlBuffer = device.newBufferWithLength(buffer.size * sizeof(Float), options: .CPUCacheModeDefaultCache)
            mtlBuffer.label = "\(buffer.name)Buffer"
            self.buffers.append(mtlBuffer)
        }
    }
    
    func processNodes(commandQueue: MTLCommandQueue, terminateForwardPass: (ForwardRunnerInstance) -> Void) {
        while !openNodes.isEmpty {
            let node = openNodes.popLast()!
            if closedNodes.contains(node) {
                continue
            }

            guard let forwardLayer = node.layer as? ForwardLayer else {
                continue
            }

            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }

            let inputBuffer = buffers[input.id]
            let outputBuffer = buffers[output.id]

            let buffer = commandQueue.commandBuffer()
            forwardLayer.encodeForwardInBuffer(buffer,
                                               batchSize: batchSize,
                                               input: inputBuffer,
                                               offset: node.inputOffset,
                                               output: outputBuffer,
                                               offset: node.outputOffset)


            buffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    self.finishNode(node)
                    if self.isFinished() {
                        terminateForwardPass(self)
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
        guard let buffer = node.inputBuffer else {
            return true
        }
        
        for n in buffer.inputNodes {
            if !closedNodes.contains(n) {
                return false
            }
        }
        return true
    }
    
    func closeNode(node: NetNode) {
        closedNodes.insert(node)
        
        if let buffer = node.outputBuffer {
            let newOpenNodes = buffer.outputNodes.filter{ isNodeReady($0) }
            openNodes.appendContentsOf(newOpenNodes)
        }
    }
    
    func finishNode(node: NetNode) {
        finishedNodes.insert(node)
    }
    
    func isFinished() -> Bool {
        return openNodes.isEmpty && closedNodes == finishedNodes
    }

    func wipeBuffers(runner: Runner) {
        let commandBuffer = runner.commandQueue.commandBuffer()
        for buffer in buffers {
            var dimensions = BufferDimensions(count: UInt32(buffer.length))
            let bufferDimensions = buffer.device.newBufferWithBytes(&dimensions, length: sizeof(BufferDimensions), options: .CPUCacheModeWriteCombined)
            bufferDimensions.label = "BufferDimensions"

            let encoder = commandBuffer.computeCommandEncoder()
            encoder.label = "ResetBuffer"
            encoder.setComputePipelineState(runner.resetState)
            encoder.setBuffer(buffer, offset: 0, atIndex: 0)
            encoder.setBuffer(bufferDimensions, offset: 0, atIndex: 1)

            let count = Int(dimensions.count)
            let threadsPerGroup = MTLSize(width: runner.resetState.threadExecutionWidth, height: 1, depth: 1)
            let numThreadgroups = MTLSize(width: (count - 1) / runner.resetState.threadExecutionWidth + 1, height: 1, depth:1)
            encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
            
            encoder.endEncoding()
        }
        commandBuffer.commit()
    }
}
