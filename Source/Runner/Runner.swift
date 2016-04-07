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

    let device: MTLDevice
    var library: MTLLibrary!
    var commandQueue: MTLCommandQueue

    let instanceCount = 3

    var forwardQueue: dispatch_queue_t
    var forwardInflightSemaphore: dispatch_semaphore_t
    var forwardInstances = [ForwardRunnerInstance]()
    var nextForwardInstanceIndex = 0
    public var forwardPassAction: ((ForwardRunnerInstance) -> Void)?

    var backwardQueue: dispatch_queue_t
    var backwardInflightSemaphore: dispatch_semaphore_t
    var backwardInstances = [BackwardRunnerInstance]()
    var nextBackwardInstanceIndex = 0
    public var backwardPassAction: ((BackwardRunnerInstance, ForwardRunnerInstance) -> Void)?

    public init(net: Net, device: MTLDevice, batchSize: Int = 1, backward: Bool = false) throws {
        self.batchSize = batchSize
        self.net = net
        self.device = device

        commandQueue = device.newCommandQueue()
        commandQueue.label = "BrainCore.Runner"
        forwardQueue = dispatch_queue_create("BrainCore.Runner Forward", DISPATCH_QUEUE_SERIAL)
        backwardQueue = dispatch_queue_create("BrainCore.Runner Backward", DISPATCH_QUEUE_SERIAL)
        forwardInflightSemaphore = dispatch_semaphore_create(instanceCount)
        backwardInflightSemaphore = dispatch_semaphore_create(instanceCount)

        guard let path = NSBundle(forClass: self.dynamicType).pathForResource("default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        library = try device.newLibraryWithFile(path)

        for _ in 0..<instanceCount {
            let forwardInstance = ForwardRunnerInstance(buffers: net.buffers, device: device, batchSize: batchSize)
            forwardInstances.append(forwardInstance)
            if backward {
                let backwardwardInstance = BackwardRunnerInstance(diffBuffers: net.buffers, device: device, batchSize: batchSize)
                backwardInstances.append(backwardwardInstance)
            }
        }

        for node in net.nodes {
            if let forwardLayer = node.layer as? ForwardLayer {
                try forwardLayer.setupInLibrary(library)
            }
        }
    }

    /// Perform a forward pass on the network
    public func forward() {
        dispatch_semaphore_wait(forwardInflightSemaphore, DISPATCH_TIME_FOREVER)

        let instance = forwardInstances[nextForwardInstanceIndex]
        instance.reset()
        nextForwardInstanceIndex = (nextForwardInstanceIndex + 1) % instanceCount

        commandQueue.insertDebugCaptureBoundary()

        // Collect all data
        for n in net.dataNodes {
            let dataLayer = n.layer as! DataLayer

            if let netBuffer = n.outputBuffer {
                let buffer = instance.buffers[netBuffer.id]
                fillBuffer(buffer, start: batchSize * n.outputOffset, withElements: dataLayer.nextBatch(batchSize))
            }
            instance.closeNode(n)
            instance.finishNode(n)
        }

        dispatch_async(forwardQueue) {
            instance.processNodes(self.commandQueue, terminateForwardPass: self.terminateForwardPass)
        }
    }

    /// Perform a backward pass on the network
    public func backward(forwardInstance: ForwardRunnerInstance) {
        dispatch_semaphore_wait(backwardInflightSemaphore, DISPATCH_TIME_FOREVER)

        let backwardInstance = backwardInstances[nextBackwardInstanceIndex]
        backwardInstance.reset()
        nextBackwardInstanceIndex = (nextBackwardInstanceIndex + 1) % instanceCount

        commandQueue.insertDebugCaptureBoundary()

        for n in net.nodes {
            if n.layer is LossLayer {
                backwardInstance.openNodes.append(n)
            }
        }

        dispatch_async(backwardQueue) {
            backwardInstance.processNodes(self.commandQueue, forwardInstance: forwardInstance, terminateBackwardPass: self.terminateBackwardPass)
        }
    }

    func terminateForwardPass(instance: ForwardRunnerInstance) {
        // Collect all data
        for n in net.sinkNodes {
            let sinkLayer = n.layer as! SinkLayer
            if let netBuffer = n.inputBuffer {
                let buffer = instance.buffers[netBuffer.id]
                sinkLayer.consume(valueArrayFromBuffer(buffer, start: n.inputOffset))
            }
        }

        self.forwardPassAction?(instance)
        dispatch_semaphore_signal(forwardInflightSemaphore);
    }

    func terminateBackwardPass(instance: BackwardRunnerInstance, forwardInstance: ForwardRunnerInstance) {
        self.backwardPassAction?(instance, forwardInstance)
        dispatch_semaphore_signal(backwardInflightSemaphore);
    }
}
