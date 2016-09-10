// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

/// A buffer that is part of a network definition.
class NetBuffer: Hashable {
    enum `Type`: Int {
        case forward
        case deltas
        case parameters
    }

    /// Buffer unique identifier.
    let id: UUID

    /// Buffer type.
    let type: Type

    /// Optional buffer name.
    let name: String?

    var inputSize: Int {
        return inputNodes.reduce(0, { currentValue, weakNode in
            return max(currentValue, weakNode.node.outputRange.upperBound)
        })
    }

    var outputSize: Int {
        return outputNodes.reduce(0, { currentValue, weakNode in
            return max(currentValue, weakNode.node.inputRange.upperBound)
        })
    }

    var size: Int {
        precondition(inputSize == outputSize, "Incompatible input (\(inputSize)) and output (\(outputSize)) sizes.")
        return inputSize
    }

    var inputNodes = [WeakNetNode]()
    var outputNodes = [WeakNetNode]()
    
    init(id: UUID, type: Type, name: String? = nil) {
        self.id = id
        self.type = type
        self.name = name
    }

    init(type: Type, name: String? = nil) {
        id = UUID()
        self.type = type
        self.name = name
    }

    var hashValue: Int {
        return id.hashValue ^ type.rawValue.hashValue
    }
}

func == (lhs: NetBuffer, rhs: NetBuffer) -> Bool {
    return lhs.id == rhs.id && lhs.type == rhs.type
}
