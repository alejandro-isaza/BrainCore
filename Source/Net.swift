// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class Net {
    public typealias BufferRef = Int
    public typealias LayerRef = Int

    var buffers = [NetBuffer]()
    var dataNodes = [NetNode]()
    var sinkNodes = [NetNode]()
    var nodes = [NetNode]()

    public init() {
    }

    public func addBufferWithName(name: String, size: Int) -> BufferRef {
        let buffer = NetBuffer(id: buffers.count, name: name, size: size)
        buffers.append(buffer)
        return buffer.id
    }

    public func addLayer(layer: Layer, name: String) -> LayerRef {
        validateLayer(layer)
        let node = NetNode(id: nodes.count, name: name, layer: layer)

        if layer is DataLayer {
            dataNodes.append(node)
        } else if layer is SinkLayer {
            sinkNodes.append(node)
        }
        nodes.append(node)

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
        for node in nodes {
            if node.name == name {
                return node.layer
            }
        }
        return nil
    }

    /// Get a layer reference by name
    public func layerRefWithName(name: String) -> LayerRef? {
        for (ref, node) in nodes.enumerate() {
            if node.name == name {
                return ref
            }
        }
        return nil
    }

    /// Get a node by layer name
    func nodeWithLayerName(name: String) -> NetNode? {
        for node in nodes {
            if node.name == name {
                return node
            }
        }
        return nil
    }

    /// Get a buffer by name
    func bufferWithName(name: String) -> NetBuffer? {
        for buffer in buffers {
            if buffer.name == name {
                return buffer
            }
        }
        return nil
    }

    /// Send the output of a layer to the given buffer with an optional offset
    public func connectLayer(layerName: String, toBuffer bufferName: String, atOffset offset: Int = 0) {
        guard let node = nodeWithLayerName(layerName) else {
            preconditionFailure("Layer not found '\(layerName)'")
        }

        guard let buffer = bufferWithName(bufferName) else {
            preconditionFailure("Buffer not found '\(bufferName)'")
        }

        node.outputBuffer = buffer
        node.outputOffset = offset
        buffer.inputNodes.append(node)
    }

    /// Send the output of a layer to the given buffer with an optional offset
    public func connectLayer(layerID: LayerRef, toBuffer bufferID: BufferRef, atOffset offset: Int = 0) {
        let node = nodes[layerID]
        let buffer = buffers[bufferID]

        node.outputBuffer = buffer
        node.outputOffset = offset
        buffer.inputNodes.append(node)
    }


    /// Get the input of a layer from the given buffer with an optional offset
    public func connectBuffer(bufferName: String, atOffset offset: Int = 0, toLayer layerName: String) {
        guard let node = nodeWithLayerName(layerName) else {
            preconditionFailure("Layer not found '\(layerName)'")
        }

        guard let buffer = bufferWithName(bufferName) else {
            preconditionFailure("Buffer not found '\(bufferName)'")
        }

        node.inputBuffer = buffer
        node.inputOffset = offset
        buffer.outputNodes.append(node)
    }

    /// Get the input of a layer from the given buffer with an optional offset
    public func connectBuffer(bufferID: BufferRef, atOffset offset: Int = 0, toLayer layerID: LayerRef) {
        let node = nodes[layerID]
        let buffer = buffers[bufferID]

        node.inputBuffer = buffer
        node.inputOffset = offset
        buffer.outputNodes.append(node)
    }
}
