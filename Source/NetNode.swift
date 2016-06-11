// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

/// A network definition node.
class NetNode: Hashable {
    let layer: Layer

    weak var inputBuffer: NetBuffer?
    var inputRange = 0...0

    weak var outputBuffer: NetBuffer?
    var outputRange = 0...0

    var inputSize: Int {
        if let forwardLayer = layer as? ForwardLayer {
            return forwardLayer.inputSize
        } else if let sinkLayer = layer as? SinkLayer {
            return sinkLayer.inputSize
        }
        return 0
    }

    var outputSize: Int {
        if let forwardLayer = layer as? ForwardLayer {
            return forwardLayer.outputSize
        } else if let dataLayer = layer as? DataLayer {
            return dataLayer.outputSize
        }
        return 0
    }

    init(layer: Layer) {
        self.layer = layer
    }

    var hashValue: Int {
        return layer.id.hashValue
    }
}

func ==(lhs: NetNode, rhs: NetNode) -> Bool {
    return lhs.layer.id == rhs.layer.id
}

class WeakNetNode {
    weak var node: NetNode!
    init(_ node: NetNode) {
        self.node = node
    }
}
