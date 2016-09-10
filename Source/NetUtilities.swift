//  Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

infix operator => : AdditionPrecedence
infix operator =>> : AdditionPrecedence

/// Adds a connection between two layers.
///
/// The output of the left-side layer will be split if there are pre-existing connections. For instance if there are two connections `A => B` and `A => C` the first `N` elements will go to `B` and the rest will go to `C` where `N` is the input size of `B`. Use the `=>>` operator if you want to send copies of the same data to multiple layers.
///
/// - Important: Use this operator only within the closure of `Net.build()`.
@discardableResult
public func =>(leftLayer: Layer, rightLayer: Layer) -> Layer {
    guard let net = Net.buildStack.last else {
        preconditionFailure("Network build operations can only happen inside `Net.build`.")
    }

    let leftNode: NetNode
    if let node = net.nodeForLayer(leftLayer) {
        leftNode = node
    } else {
        net.addLayer(leftLayer)
        leftNode = net.nodeForLayer(leftLayer)!
    }

    let rightNode: NetNode
    if let node = net.nodeForLayer(rightLayer) {
        rightNode = node
    } else {
        net.addLayer(rightLayer)
        rightNode = net.nodeForLayer(rightLayer)!
    }

    if let buffer = leftNode.outputBuffer {
        precondition(rightNode.inputBuffer == nil || rightNode.inputBuffer! == buffer, "Can't rewire layers that are already connected")
        net.connectSplitBuffer(buffer, toNode: rightNode)
    } else if let buffer = rightNode.inputBuffer {
        precondition(leftNode.outputBuffer == nil || leftNode.outputBuffer! == buffer, "Can't rewire layers that are already connected")
        net.connectNode(leftNode, toBuffer: buffer)
    } else {
        let id = net.addBuffer()
        let buffer = net.bufferWithID(id)!
        net.connectNode(leftNode, toBuffer: buffer)
        net.connectSplitBuffer(buffer, toNode: rightNode)
    }

    return rightLayer
}

/// Adds a connection between an array of layers and a target layer by concatenating the output of the layers in the array.
///
/// - Important: Use this operator only within the closure of `Net.build()`.
@discardableResult
public func =>(leftLayers: [Layer], rightLayer: Layer) -> Layer {
    guard let net = Net.buildStack.last else {
        preconditionFailure("Network build operations can only happen inside `Net.build`.")
    }

    var nodes = [NetNode]()
    var buffer: NetBuffer? = nil
    for layer in leftLayers {
        net.addLayer(layer)
        let node = net.nodeForLayer(layer)!

        if let existingBuffer = node.outputBuffer {
            precondition(buffer == nil || existingBuffer == buffer!, "Can't rewire layers that are already connected")
            buffer = existingBuffer
        }
        nodes.append(node)
    }

    if buffer == nil {
        buffer = net.bufferWithID(net.addBuffer())
    }

    for node in nodes {
        net.connectNode(node, toBuffer: buffer!)
    }
    net.connectSplitBuffer(buffer!.id, toLayer: rightLayer)

    return rightLayer
}

/// Adds a connection between a layer and an array of layers by splitting the output of the layer.
///
/// - Important: Use this operator only within the closure of `Net.build()`.
@discardableResult
public func =>(leftLayer: Layer, rightLayers: [Layer]) -> [Layer] {
    guard let net = Net.buildStack.last else {
        preconditionFailure("Network build operations can only happen inside `Net.build`.")
    }

    var nodes = [NetNode]()
    var buffer: NetBuffer? = nil
    for layer in rightLayers {
        net.addLayer(layer)
        let node = net.nodeForLayer(layer)!
        if let existingBuffer = node.inputBuffer {
            precondition(buffer == nil || existingBuffer == buffer!, "Can't rewire layers that are already connected")
            buffer = existingBuffer
        }
        nodes.append(node)
    }

    if buffer == nil {
        buffer = net.bufferWithID(net.addBuffer())
    }

    for node in nodes {
        net.connectSplitBuffer(buffer!, toNode: node)
    }
    net.connectLayer(leftLayer, toBuffer: buffer!.id)

    return rightLayers
}

/// Adds a connection between two layers without splits.
///
/// As opposed to `=>`, `=>>` will not split the result of the left-hand layer. Instead it will send a copy starting at element `0` to the right-hand layer.
///
/// - Important: Use this operator only within the closure of `Net.build()`.
@discardableResult
public func =>>(leftLayer: Layer, rightLayer: Layer) -> Layer {
    guard let net = Net.buildStack.last else {
        preconditionFailure("Network build operations can only happen inside `Net.build`.")
    }

    let leftNode: NetNode
    if let node = net.nodeForLayer(leftLayer) {
        leftNode = node
    } else {
        net.addLayer(leftLayer)
        leftNode = net.nodeForLayer(leftLayer)!
    }

    let rightNode: NetNode
    if let node = net.nodeForLayer(rightLayer) {
        rightNode = node
    } else {
        net.addLayer(rightLayer)
        rightNode = net.nodeForLayer(rightLayer)!
    }

    if let buffer = leftNode.outputBuffer {
        precondition(rightNode.inputBuffer == nil || rightNode.inputBuffer! == buffer, "Can't rewire layers that are already connected")
        net.connectWholeBuffer(buffer, toNode: rightNode)
    } else if let _ = rightNode.inputBuffer {
        preconditionFailure("Can't rewire layers that are already connected")
    } else {
        let id = net.addBuffer()
        let buffer = net.bufferWithID(id)!
        net.connectNode(leftNode, toBuffer: buffer)
        net.connectWholeBuffer(buffer, toNode: rightNode)
    }

    return rightLayer
}
