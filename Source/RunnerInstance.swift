// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

class RunnerInstance {
    var buffers: [MTLBuffer]
    var openNodes = [NetNode]()
    var closedNodes = Set<NetNode>()
    var finishedNodes = Set<NetNode>()

    init(buffers: [NetBuffer], device: MTLDevice) {
        self.buffers = [MTLBuffer]()
        self.buffers.reserveCapacity(buffers.count)
        for buffer in buffers {
            let mtlBuffer = device.newBufferWithLength(buffer.size * sizeof(Float), options: .CPUCacheModeDefaultCache)
            mtlBuffer.label = "\(buffer.name)Buffer"
            self.buffers.append(mtlBuffer)
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
}
