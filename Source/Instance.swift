// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

/// Helper class to keep track of the state of each node in the network for a particular run.
class Instance {
    let batchSize: Int

    var buffers: [UUID: MTLBuffer]

    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    init(buffers: [UUID: NetBuffer], device: MTLDevice, batchSize: Int) {
        self.batchSize = batchSize

        self.buffers = [UUID: MTLBuffer]()

        for (id, buffer) in buffers {
            let mtlForwardBuffer = device.makeBuffer(length: buffer.size * batchSize * MemoryLayout<Float>.size, options: MTLResourceOptions())
            mtlForwardBuffer.label = "\(buffer.name)Buffer"
            self.buffers[id] = mtlForwardBuffer
        }
    }

    func isOpen(_ node: NetNode) -> Bool {
        return openNodes.contains(node)
    }

    func isClosed(_ node: NetNode) -> Bool {
        return closedNodes.contains(node)
    }

    func isFinished() -> Bool {
        return openNodes.isEmpty && closedNodes == finishedNodes
    }

    func reset() {
        openNodes.removeAll(keepingCapacity: true)
        closedNodes.removeAll(keepingCapacity: true)
        finishedNodes.removeAll(keepingCapacity: true)
    }

    func openOutputsOf(_ node: NetNode) {
        guard let buffer = node.outputBuffer else {
            return
        }

        let newOpenNodes = buffer.outputNodes.lazy.map({ $0.node }).filter(allInputsClosed)
        openNodes.append(contentsOf: newOpenNodes)
    }

    func allInputsClosed(_ node: NetNode) -> Bool {
        guard let buffer = node.inputBuffer else {
            return true
        }

        for n in buffer.inputNodes {
            if !isClosed(n.node) {
                return false
            }
        }
        return true
    }

    func openInputsOf(_ node: NetNode) {
        guard let buffer = node.inputBuffer else {
            return
        }

        let newOpenNodes = buffer.inputNodes.lazy.map({ $0.node }).filter(allOutputsClosed)
        openNodes.append(contentsOf: newOpenNodes)
    }

    func allOutputsClosed(_ node: NetNode) -> Bool {
        guard let buffer = node.outputBuffer else {
            return true
        }

        for n in buffer.outputNodes {
            if !isClosed(n.node) {
                return false
            }
        }
        return true
    }

    func closeNode(_ node: NetNode) {
        closedNodes.insert(node)
    }

    func finishNode(_ node: NetNode) {
        finishedNodes.insert(node)
    }
}
