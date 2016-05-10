// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

struct GPUBufferHeader {
    var inputSize: UInt32
    var sequenceSize: UInt32
    var batchSize: UInt32

    init(inputSize: Int, sequenceSize: Int, batchSize: Int) {
        self.inputSize = UInt32(inputSize)
        self.sequenceSize = UInt32(sequenceSize)
        self.batchSize = UInt32(batchSize)
    }
}


/// Keeps track of the state of each node in the network for a particular training batch.
class Instance {
    let batchSize: Int

    var buffers: [NSUUID: MTLBuffer]

    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    init(buffers: [NSUUID: NetBuffer], device: MTLDevice, batchSize: Int) {
        self.batchSize = batchSize

        self.buffers = [NSUUID: MTLBuffer]()

        for (id, buffer) in buffers {
            let size = sizeof(GPUBufferHeader) + buffer.size * batchSize * sizeof(Float)
            let metalBuffer = device.newBufferWithLength(size, options: .CPUCacheModeDefaultCache)
            metalBuffer.label = "\(buffer.name)Buffer"
            self.buffers[id] = metalBuffer

            let header = GPUBufferHeader(inputSize: buffer.size, sequenceSize: 1, batchSize: batchSize)
            UnsafeMutablePointer<GPUBufferHeader>(metalBuffer.contents()).memory = header
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
