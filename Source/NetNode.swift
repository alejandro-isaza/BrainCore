// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

class NetNode: Hashable {
    let id: Int
    let layer: Layer
    let name: String

    weak var inputBuffer: NetBuffer?
    var inputOffset = 0

    weak var outputBuffer: NetBuffer?
    var outputOffset = 0

    init(id: Int, name: String, layer: Layer) {
        self.id = id
        self.name = name
        self.layer = layer
    }

    var hashValue: Int {
        return id
    }
}

func ==(lhs: NetNode, rhs: NetNode) -> Bool {
    return lhs.id == rhs.id
}
