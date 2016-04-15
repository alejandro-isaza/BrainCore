// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

class NetNode: Hashable {
    let layer: Layer

    weak var inputBuffer: NetBuffer?
    var inputOffset = 0

    weak var outputBuffer: NetBuffer?
    var outputOffset = 0

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
