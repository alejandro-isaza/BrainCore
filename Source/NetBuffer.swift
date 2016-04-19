// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.


class NetBuffer: Hashable {
    /// Buffer unique identifier
    let id = NSUUID()

    /// Optional buffer name
    let name: String?

    var inputSize: Int {
        return inputNodes.reduce(0, combine: { currentValue, node in
            return max(currentValue, node.outputRange.endIndex)
        })
    }

    var outputSize: Int {
        return outputNodes.reduce(0, combine: { currentValue, node in
            return max(currentValue, node.inputRange.endIndex)
        })
    }

    var size: Int {
        precondition(inputSize == outputSize, "Incompatible input (\(inputSize)) and output (\(outputSize)) sizes.")
        return inputSize
    }

    var inputNodes = [NetNode]()
    var outputNodes = [NetNode]()
    
    init(name: String? = nil) {
        self.name = name
    }

    var hashValue: Int {
        return id.hashValue
    }
}

func == (lhs: NetBuffer, rhs: NetBuffer) -> Bool {
    return lhs.id == rhs.id
}
