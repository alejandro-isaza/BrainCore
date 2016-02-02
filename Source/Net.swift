// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class Net {
    public typealias LayerRef = Int

    class Node : Hashable {
        let id: Int
        let layer: Layer
        let name: String
        var inputNodes = [Node]()
        var outputNodes = [Node]()
        var input: MTLBuffer?
        var output: MTLBuffer?

        init(layer: Layer, name: String, id: Int) {
            self.id = id
            self.layer = layer
            self.name = name
        }

        var hashValue: Int {
            return id
        }
    }

    var nextID = 0
    var dataNodes = [Node]()
    var nodes = [LayerRef: Node]()

    var openNodes = [Node]()
    var closedNodes = Set<Node>()

    var device: MTLDevice
    var library: MTLLibrary
    var commandQueue: MTLCommandQueue

    var queue: dispatch_queue_t
    var activeThreads = 0

    public init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
        commandQueue = device.newCommandQueue()
        queue = dispatch_queue_create("Net", DISPATCH_QUEUE_SERIAL)
    }

    public func addLayer(layer: Layer, name: String) -> LayerRef {
        validateLayer(layer)
        let node = Node(layer: layer, name: name, id: nextID++)

        if layer is DataLayer {
            dataNodes.append(node)
        }
        nodes[node.id] = node

        return node.id
    }

    private func validateLayer(layer: Layer) {
        let dataLayer = layer is DataLayer
        let forwardLayer = layer is ForwardLayer
        let sinkLayer = layer is SinkLayer
        precondition(dataLayer || forwardLayer || sinkLayer, "Layer has to be one of: data, forward, sink")
        precondition(!dataLayer || !forwardLayer, "Layer can't be both a data layer and a forward layer")
        precondition(!sinkLayer || !forwardLayer, "Layer can't be both a sink layer and a forward layer")
        precondition(!dataLayer || !sinkLayer, "Layer can't be both a data layer and a sink layer")
    }

    /// Find a layer by name
    public func layerWithName(name: String) -> Layer? {
        for (_, node) in nodes {
            if node.name == name {
                return node.layer
            }
        }
        return nil
    }

    /// Get a layer reference by name
    public func layerRefWithName(name: String) -> LayerRef? {
        for (ref, node) in nodes {
            if node.name == name {
                return ref
            }
        }
        return nil
    }

    /// Get a node by layer name
    func nodeWithLayerName(name: String) -> Node? {
        for (_, node) in nodes {
            if node.name == name {
                return node
            }
        }
        return nil
    }

    /// Connect two layers given their names
    public func connectLayer(layer1: String, toLayer layer2: String) {
        guard let ref1 = layerRefWithName(layer1) else {
            precondition(false, "Layer not found '\(layer1)'")
            return
        }
        guard let ref2 = layerRefWithName(layer2) else {
            precondition(false, "Layer not found '\(layer1)'")
            return
        }
        connectLayer(ref1, toLayer: ref2)
    }

    /// Connect two layers
    public func connectLayer(layer1: LayerRef, toLayer layer2: LayerRef) {
        guard let node1 = nodes[layer1] else {
            precondition(false, "Layer not found")
            return
        }
        guard let node2 = nodes[layer2] else {
            precondition(false, "Layer not found")
            return
        }

        node1.outputNodes.append(node2)
        node2.inputNodes.append(node1)
    }

    /// Get the input data that was used for a layer on the last forward pass
    public func lastLayerInput(layerName: String) -> Array<Float>? {
        guard let node = nodeWithLayerName(layerName) else {
            return nil
        }
        guard let input = node.input else {
            return nil
        }
        return arrayFromBuffer(input)
    }

    /// Get the output data of a layer on the last forward pass
    public func lastLayerOutput(layerName: String) -> Array<Float>? {
        guard let node = nodeWithLayerName(layerName) else {
            return nil
        }
        guard let output = node.output else {
            return nil
        }
        return arrayFromBuffer(output)
    }

    /// Perform a forward pass on the network
    public func forward(completion completion: (() -> Void)?) {
        precondition(activeThreads == 0, "You can only run one forward pass at a time")

        openNodes.removeAll(keepCapacity: true)
        closedNodes.removeAll(keepCapacity: true)

        // Collect all data
        for n in dataNodes {
            let dataLayer = n.layer as! DataLayer
            if let buffer = n.output where buffer.length / sizeof(Float) == dataLayer.data.count {
                fillBuffer(buffer, withElements: dataLayer.data)
            } else {
                let buffer = device.newBufferWithBytes(dataLayer.data.pointer, length: dataLayer.data.count * sizeof(Float), options: .CPUCacheModeWriteCombined)
                fillBuffer(buffer, withElements: dataLayer.data)
                n.output = buffer
            }
            closeNode(n)
        }

        dispatch_async(queue) {
            self.processNodes(completion)
        }
    }

    private func processNodes(completion: (() -> Void)?) {
        while !openNodes.isEmpty {
            let node = openNodes.popLast()!
            if closedNodes.contains(node) {
                continue
            }

            let data = collectDataForNode(node)
            if let forwardLayer = node.layer as? ForwardLayer {
                let outputBuffer = setupOutputData(node, size: forwardLayer.outputSize)

                let commandBuffer = commandQueue.commandBuffer()
                forwardLayer.encodeForwardInBuffer(commandBuffer, input: data, output: outputBuffer)
                commandBuffer.addCompletedHandler() { commandBuffer in
                    dispatch_async(self.queue) {
                        self.activeThreads -= 1
                        self.closeNode(node)
                        self.processNodes(completion)
                    }
                }
                commandBuffer.commit()
                activeThreads += 1
            } else if let sinkLayer = node.layer as? SinkLayer {
                sinkLayer.consume(valueArrayFromBuffer(data))
            }
        }

        if activeThreads == 0 {
            completion?()
        }
    }

    private func closeNode(node: Node) {
        closedNodes.insert(node)
        let newOpenNodes = node.outputNodes.filter{ isNodeReady($0) }
        openNodes.appendContentsOf(newOpenNodes)
    }

    private func isNodeReady(node: Node) -> Bool {
        for n in node.inputNodes {
            if !closedNodes.contains(n) {
                return false
            }
        }
        return true
    }

    private func collectDataForNode(node: Node) -> MTLBuffer {
        var size = 0
        for n in node.inputNodes {
            size += n.output?.length ?? 0
        }

        let data: MTLBuffer
        if let input = node.input where input.length >= size {
            data = input
        } else {
            data = device.newBufferWithLength(size , options: .CPUCacheModeWriteCombined)
            node.input = data
        }

        let pointer = UnsafeMutablePointer<Float>(data.contents())
        var i = 0
        for n in node.inputNodes {
            if let output = n.output {
                let outputBuffer = unsafeBufferPointerFromBuffer(output)
                (pointer + i).assignFrom(outputBuffer.baseAddress, count: outputBuffer.count)
                i += outputBuffer.count
            }
        }

        return data
    }

    private func setupOutputData(node: Node, size: Int) -> MTLBuffer {
        let data: MTLBuffer
        if let output = node.output where output.length >= size * sizeof(Float) {
            data = output
        } else {
            data = device.newBufferWithLength(size * sizeof(Float), options: .CPUCacheModeDefaultCache)
            node.output = data
        }
        return data
    }
}

func == (lhs: Net.Node, rhs: Net.Node) -> Bool {
    return lhs.id == rhs.id
}
