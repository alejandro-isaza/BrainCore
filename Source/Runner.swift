// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class Runner {
    public let batchSize: Int
    public let net: Net
    public var forwardPassAction: (([MTLBuffer]) -> Void)?

    let device: MTLDevice
    var library: MTLLibrary!
    var commandQueue: MTLCommandQueue

    var queue: dispatch_queue_t
    var inflightSemaphore: dispatch_semaphore_t

    let instanceCount = 3
    var instances = [RunnerInstance]()
    var nextInstanceIndex = 0

    public init(net: Net, device: MTLDevice, batchSize: Int = 1) throws {
        self.batchSize = batchSize
        self.net = net
        self.device = device

        commandQueue = device.newCommandQueue()
        commandQueue.label = "BrainCore.Runner"
        queue = dispatch_queue_create("BrainCore.Runner", DISPATCH_QUEUE_SERIAL)
        inflightSemaphore = dispatch_semaphore_create(instanceCount)

        guard let path = NSBundle(forClass: self.dynamicType).pathForResource("default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        library = try device.newLibraryWithFile(path)

        for _ in 0..<instanceCount {
            let instance = RunnerInstance(buffers: net.buffers, device: device, batchSize: batchSize)
            instances.append(instance)
        }

        for node in net.nodes {
            if let forwardLayer = node.layer as? ForwardLayer {
                try forwardLayer.setupInLibrary(library)
            }
        }
    }

    /// Perform a forward pass on the network
    public func forward() {
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)

        let instance = instances[nextInstanceIndex]
        instance.reset()
        nextInstanceIndex = (nextInstanceIndex + 1) % instanceCount

        commandQueue.insertDebugCaptureBoundary()

        // Collect all data
        for n in net.dataNodes {
            let dataLayer = n.layer as! DataLayer
            precondition(dataLayer.data.count == dataLayer.outputSize * batchSize)

            if let netBuffer = n.outputBuffer {
                let buffer = instance.buffers[netBuffer.id]
                fillBuffer(buffer, start: n.outputOffset, withElements: dataLayer.data)
            }
            instance.closeNode(n)
            instance.finishNode(n)
        }

        dispatch_async(queue) {
            self.processNodesOfInstance(instance)
        }
    }

    func processNodesOfInstance(instance: RunnerInstance) {
        while !instance.openNodes.isEmpty {
            let node = instance.openNodes.popLast()!
            if instance.closedNodes.contains(node) {
                continue
            }

            guard let forwardLayer = node.layer as? ForwardLayer else {
                continue
            }

            guard let input = node.inputBuffer, output = node.outputBuffer else {
                preconditionFailure("Layer '\(node.name)' is missing a buffer")
            }

            let inputBuffer = instance.buffers[input.id]
            let outputBuffer = instance.buffers[output.id]

            let commandBuffer = commandQueue.commandBuffer()
            forwardLayer.encodeForwardInBuffer(commandBuffer,
                batchSize: batchSize, input: inputBuffer,
                offset: node.inputOffset, output: outputBuffer,
                offset: node.outputOffset)

            commandBuffer.addCompletedHandler() { commandBuffer in
                dispatch_async(self.queue) {
                    instance.finishNode(node)
                    if instance.isFinished() {
                        self.terminateForwardPass(instance)
                    }
                }
            }
            commandBuffer.commit()

            instance.closeNode(node)
        }
    }

    func terminateForwardPass(instance: RunnerInstance) {

        // Collect all data
        for n in net.sinkNodes {
            let sinkLayer = n.layer as! SinkLayer
            if let netBuffer = n.inputBuffer {
                let buffer = instance.buffers[netBuffer.id]
                sinkLayer.consume(valueArrayFromBuffer(buffer, start: n.inputOffset))
            }
        }

        self.forwardPassAction?(instance.buffers)
        dispatch_semaphore_signal(inflightSemaphore);
    }
}
