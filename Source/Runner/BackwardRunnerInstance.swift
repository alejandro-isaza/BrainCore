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
    
    var solverParametersBuffer: MTLBuffer

    init(diffBuffers: [NetBuffer], device: MTLDevice, inout solverParameters: SolverParameters) {
        self.diffBuffers = [MTLBuffer]()
        self.diffBuffers.reserveCapacity(diffBuffers.count)
        for buffer in diffBuffers {
            let mtlBuffer = device.newBufferWithLength(buffer.size * sizeof(Float), options: .CPUCacheModeDefaultCache)
            mtlBuffer.label = "\(buffer.name)Buffer"
            self.diffBuffers.append(mtlBuffer)
        }
        self.solverParametersBuffer = device.newBufferWithBytes(&solverParameters, length: sizeof(SolverParameters), options: .CPUCacheModeWriteCombined)
        self.solverParametersBuffer.label = "SolverParametersBuffer"
    }
    
    func processNodes(runner: Runner, forwardInstance: ForwardRunnerInstance) {
        while !openNodes.isEmpty {
            let node = openNodes.popLast()!
            if closedNodes.contains(node) {
                continue
            }
            
            guard let backwardLayer = node.layer as? BackwardLayer else {
                continue
            }
            
            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }
            
            let inputBuffer = forwardInstance.buffers[input.id]
            let inputDiffBuffer = diffBuffers[input.id]
            let outputDiffBuffer = diffBuffers[output.id]
            
            let commandBuffer = runner.commandQueue.commandBuffer()
            backwardLayer.encodeBackwardInBuffer(commandBuffer,
                                                 batchSize: runner.batchSize,
                                                 outputDiff: outputDiffBuffer,
                                                 input: inputBuffer,
                                                 inputDiff: inputDiffBuffer)

            
            commandBuffer.addCompletedHandler() { commandBuffer in
                dispatch_async(runner.queue) {
                    self.finishNode(node)
                    if self.isFinished() {
                        runner.terminateBackwardPass(self)
                    }
                }
            }
            commandBuffer.commit()
            
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
        
        if let buffer = node.outputBuffer {
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
