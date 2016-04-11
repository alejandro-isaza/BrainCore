// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

/// Keeps track of the state of each node in the network for a particular training batch.
class Instance {
    let batchSize: Int

    var buffers: [MTLBuffer]

    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    init(buffers: [NetBuffer], device: MTLDevice, batchSize: Int) {
        self.batchSize = batchSize

        self.buffers = [MTLBuffer]()
        self.buffers.reserveCapacity(buffers.count)

        for buffer in buffers {
            let mtlForwardBuffer = device.newBufferWithLength(buffer.size * batchSize * sizeof(Float), options: .CPUCacheModeDefaultCache)
            mtlForwardBuffer.label = "\(buffer.name)Buffer"
            self.buffers.append(mtlForwardBuffer)
        }
    }

    func isOpen(node: NetNode) -> Bool {
        return openNodes.contains(node)
    }

    func isClosed(node: NetNode) -> Bool {
        return closedNodes.contains(node)
    }

    func isFinished() -> Bool {
        return openNodes.isEmpty && closedNodes == finishedNodes
    }

    func reset() {
        openNodes.removeAll(keepCapacity: true)
        closedNodes.removeAll(keepCapacity: true)
        finishedNodes.removeAll(keepCapacity: true)
    }

    func openOutputsOf(node: NetNode) {
        guard let buffer = node.outputBuffer else {
            return
        }

        let newOpenNodes = buffer.outputNodes.filter(allInputsClosed)
        openNodes.appendContentsOf(newOpenNodes)
    }

    func allInputsClosed(node: NetNode) -> Bool {
        guard let buffer = node.inputBuffer else {
            return true
        }

        for n in buffer.inputNodes {
            if !isClosed(n) {
                return false
            }
        }
        return true
    }

    func openInputsOf(node: NetNode) {
        guard let buffer = node.inputBuffer else {
            return
        }

        let newOpenNodes = buffer.inputNodes.filter(allOutputsClosed)
        openNodes.appendContentsOf(newOpenNodes)
    }

    func allOutputsClosed(node: NetNode) -> Bool {
        guard let buffer = node.outputBuffer else {
            return true
        }

        for n in buffer.outputNodes {
            if !isClosed(n) {
                return false
            }
        }
        return true
    }

    func closeNode(node: NetNode) {
        closedNodes.insert(node)
    }

    func finishNode(node: NetNode) {
        finishedNodes.insert(node)
    }
}
