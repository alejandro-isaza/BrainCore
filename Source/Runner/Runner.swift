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
    public var forwardPassAction: ((ForwardRunnerInstance) -> Void)?
    public var backwardPassAction: ((BackwardRunnerInstance) -> Void)?

    let device: MTLDevice
    var library: MTLLibrary!
    var commandQueue: MTLCommandQueue

    var queue: dispatch_queue_t
    var inflightSemaphore: dispatch_semaphore_t

    let instanceCount = 3
    var forwardInstances = [ForwardRunnerInstance]()
    var nextForwardInstanceIndex = 0
    var backwardInstances = [BackwardRunnerInstance]()
    var nextBackwardInstanceIndex = 0

    public init(net: Net, device: MTLDevice, batchSize: Int = 1, params: SolverParameters? = nil) throws {
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
            let forwardInstance = ForwardRunnerInstance(buffers: net.buffers, device: device)
            forwardInstances.append(forwardInstance)
            if var params = params {
                let backwardwardInstance = BackwardRunnerInstance(diffBuffers: net.buffers, device: device, solverParameters: &params)
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
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)

        let instance = forwardInstances[nextForwardInstanceIndex]
        instance.reset()
        nextForwardInstanceIndex = (nextForwardInstanceIndex + 1) % instanceCount

        commandQueue.insertDebugCaptureBoundary()

        // Collect all data
        for n in net.dataNodes {
            let dataLayer = n.layer as! DataLayer
            precondition(dataLayer.data.count == dataLayer.outputSize)

            if let netBuffer = n.outputBuffer {
                precondition(dataLayer.data.count <= netBuffer.size - n.outputOffset)
                let buffer = instance.buffers[netBuffer.id]
                fillBuffer(buffer, start: n.outputOffset, withElements: dataLayer.data)
            }
            instance.closeNode(n)
            instance.finishNode(n)
        }

        dispatch_async(queue) {
            instance.processNodes(self)
        }
    }

    /// Perform a backward pass on the network
    public func backward(forwardInstance: ForwardRunnerInstance) {
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)
        
        let backwardInstance = backwardInstances[nextBackwardInstanceIndex]
        backwardInstance.reset()
        nextBackwardInstanceIndex = (nextBackwardInstanceIndex + 1) % instanceCount
        
        commandQueue.insertDebugCaptureBoundary()
        
        for n in net.lossNodes {
            backwardInstance.openNodes.append(n)
        }
        
        dispatch_async(queue) {
            backwardInstance.processNodes(self, forwardInstance: forwardInstance)
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
        dispatch_semaphore_signal(inflightSemaphore);
    }

    func terminateBackwardPass(instance: BackwardRunnerInstance) {
        self.backwardPassAction?(instance)
        dispatch_semaphore_signal(self.inflightSemaphore);
    }
}
