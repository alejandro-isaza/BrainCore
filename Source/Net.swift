// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

/// Neural network definition.
open class Net {
    public typealias BufferID = UUID
    typealias LayerID = UUID

    var buffers = [BufferID: NetBuffer]()
    var dataNodes = [LayerID: NetNode]()
    var lossNodes = [LayerID: NetNode]()
    var sinkNodes = [LayerID: NetNode]()
    var nodes = [LayerID: NetNode]()

    static var buildStack = [Net]()

    /// Creates a network definition with a simplified syntax using overloaded operators.
    static public func build(_ definition: () -> Void) -> Net {
        buildStack.append(Net())
        definition()
        return buildStack.popLast()!
    }

    /// Creates an empty network definition.
    public init() {
    }

    /// Creates a buffer and adds it to `self`.
    ///
    /// - Returns: The newly created buffer's identifier.
    open func addBuffer(name: String? = nil) -> BufferID {
        let buffer = NetBuffer(type: .forward, name: name)
        buffers[buffer.id] = buffer
        return buffer.id
    }

    /// Adds a layer to `self`.
    open func addLayer(_ layer: Layer) {
        if nodes[layer.id as Net.LayerID] != nil {
            return
        }

        validateLayer(layer)

        let node = NetNode(layer: layer)
        nodes[layer.id as Net.LayerID] = node

        if layer is DataLayer {
            dataNodes[layer.id as Net.LayerID] = node
        } else if layer is LossLayer {
            lossNodes[layer.id as Net.LayerID] = node
        } else if layer is SinkLayer {
            sinkNodes[layer.id as Net.LayerID] = node
        }
    }

    /// Inserts transposition layers after every `DataLayer`.
    func insertTransposeLayers() {
        for dataNode in dataNodes.values {
            let dataLayer = dataNode.layer as! DataLayer

            guard let dataOutputBuffer = dataNode.outputBuffer else {
                preconditionFailure("Layer '\(dataLayer)'s output buffer not connected.")
            }
            if dataOutputBuffer.outputNodes.map({ $0.node.layer is TransposeLayer }) == dataOutputBuffer.outputNodes.map({ _ in true }) {
                continue
            }

            let transposeBufferId = addBuffer(name: "\(dataLayer) -> Transpose")
            let transposeBuffer = buffers[transposeBufferId]!

            let transposeLayer = TransposeLayer(size: dataLayer.outputSize, name: "Transpose \(dataLayer)")
            addLayer(transposeLayer)
            let transposeNode = nodes[transposeLayer.id as Net.LayerID]!
            transposeNode.outputBuffer = dataNode.outputBuffer
            transposeNode.outputRange = dataNode.outputRange

            let dataNodeIndex = dataOutputBuffer.inputNodes.lazy.map({ $0.node }).index(of: dataNode)!
            dataOutputBuffer.inputNodes.remove(at: dataNodeIndex)

            connectNode(dataNode, toBuffer: transposeBuffer)
            connectWholeBuffer(transposeBuffer, toNode: transposeNode)

            dataOutputBuffer.inputNodes.append(WeakNetNode(transposeNode))
        }
    }

    /// Validates that a layer is one of the supported types and doesn't try to act as more than one type.
    fileprivate func validateLayer(_ layer: Layer) {
        let dataLayer = layer is DataLayer
        let forwardLayer = layer is ForwardLayer
        let sinkLayer = layer is SinkLayer
        precondition(dataLayer || forwardLayer || sinkLayer, "Layer has to be one of: data, forward, sink")
        precondition(!dataLayer || !forwardLayer, "Layer can't be both a data layer and a forward layer")
        precondition(!sinkLayer || !forwardLayer, "Layer can't be both a sink layer and a forward layer")
        precondition(!dataLayer || !sinkLayer, "Layer can't be both a data layer and a sink layer")
    }

    func nodeForLayer(_ layer: Layer) -> NetNode? {
        return nodes[layer.id as Net.LayerID]
    }

    func bufferWithID(_ id: BufferID) -> NetBuffer? {
        return buffers[id]
    }

    
    // MARK: - Net construction

    /// Sends the output of a layer to a buffer.
    open func connectLayer(_ layer: Layer, toBuffer bufferID: BufferID) {
        let node: NetNode
        if let existingNode = nodes[layer.id as Net.LayerID] {
            node = existingNode
        } else {
            addLayer(layer)
            node = nodes[layer.id as Net.LayerID]!
        }

        guard let buffer = buffers[bufferID] else {
            preconditionFailure("Could not find buffer with id: \(bufferID)")
        }
        connectNode(node, toBuffer: buffer)
    }

    func connectNode(_ node: NetNode, toBuffer buffer: NetBuffer) {
        node.outputBuffer = buffer
        node.outputRange = buffer.inputSize..<buffer.inputSize + node.outputSize
        buffer.inputNodes.append(WeakNetNode(node))
    }

    /// Sends the split contents of a buffer to a layer.
    ///
    /// If the buffer already has outgoing connections taking `N` elements this new connection will start with element `N+1`.
    ///
    /// - SeeAlso: `connectWholeBuffer(bufferID:toLayer:)`
    open func connectSplitBuffer(_ bufferID: BufferID, toLayer layer: Layer) {
        guard let buffer = buffers[bufferID] else {
            preconditionFailure("Could not find buffer with id: \(bufferID)")
        }

        let node: NetNode
        if let existingNode = nodes[layer.id as Net.LayerID] {
            node = existingNode
        } else {
            addLayer(layer)
            node = nodes[layer.id as Net.LayerID]!
        }

        connectSplitBuffer(buffer, toNode: node)
    }

    func connectSplitBuffer(_ buffer: NetBuffer, toNode node: NetNode) {
        node.inputBuffer = buffer
        node.inputRange = buffer.outputSize..<buffer.outputSize + node.inputSize
        buffer.outputNodes.append(WeakNetNode(node))
    }

    /// Sends the whole contents of a buffer to a layer.
    ///
    /// Whether or not the buffer already has outgoing connections this new connection will start with element `0`.
    ///
    /// - SeeAlso: `connectSplitBuffer(bufferID:toLayer:)`
    open func connectWholeBuffer(_ bufferID: BufferID, toLayer layer: Layer) {
        guard let buffer = buffers[bufferID] else {
            preconditionFailure("Could not find buffer with id: \(bufferID)")
        }

        let node: NetNode
        if let existingNode = nodes[layer.id as Net.LayerID] {
            node = existingNode
        } else {
            addLayer(layer)
            node = nodes[layer.id as Net.LayerID]!
        }

        connectWholeBuffer(buffer, toNode: node)
    }

    func connectWholeBuffer(_ buffer: NetBuffer, toNode node: NetNode) {
        node.inputBuffer = buffer
        node.inputRange = 0..<node.inputSize
        buffer.outputNodes.append(WeakNetNode(node))
    }
}
