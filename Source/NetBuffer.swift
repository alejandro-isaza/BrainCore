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
        return inputNodes.reduce(0) { currentValue, node in
            if let forwardLayer = node.layer as? ForwardLayer {
                return currentValue + forwardLayer.outputSize
            } else if let dataLayer = node.layer as? DataLayer {
                return currentValue + dataLayer.outputSize
            }
            preconditionFailure("Cannot costruct buffer from \(node.layer.dynamicType).")
        }
    }

    var outputSize: Int {
        return outputNodes.reduce(0) { currentValue, node in
            if let forwardLayer = node.layer as? ForwardLayer {
                return currentValue + forwardLayer.inputSize
            } else if let sinkLayer = node.layer as? SinkLayer {
                return currentValue + sinkLayer.inputSize
            }
            preconditionFailure("Cannot costruct buffer from \(node.layer.dynamicType).")
        }
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
