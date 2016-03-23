// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

struct BufferDimensions {
    let count: UInt32
}

class NetBuffer: Hashable {
    let id: Int
    let name: String
    let size: Int

    var inputNodes = [NetNode]()
    var outputNodes = [NetNode]()
    
    init(id: Int, name: String, size: Int) {
        self.id = id
        self.name = name
        self.size = size
    }

    var hashValue: Int {
        return id
    }
}

func == (lhs: NetBuffer, rhs: NetBuffer) -> Bool {
    return lhs.id == rhs.id
}
