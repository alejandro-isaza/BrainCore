// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Upsurge
import Metal

public class RNNLayer: TrainableLayer, BackwardLayer {
    public let id = UUID()
    public let name: String?

    /// The cell copies for each timestep
    public var cells: [LSTMNodeLayer]

    public var outputSize: Int {
        return cells.count * cells[0].outputSize
    }

    public var inputSize: Int {
        return cells.count * cells[0].inputSize
    }

    public var forwardInvocations: [Invocation] {
        return cells.flatMap({ $0.forwardInvocations })
    }

    public var backwardInvocations: [Invocation] {
        return cells.reversed().flatMap({ $0.backwardInvocations })
    }


    public init(weights: Matrix<Float>, biases: ValueArray<Float>, sequenceLength: Int, name: String? = nil, clipTo: Float? = nil) {
        self.name = name

        self.cells = (0..<sequenceLength).map({ LSTMNodeLayer(weights: weights, biases: biases, time: $0, name: "\(name) T\($0)", clipTo: clipTo) })

        for (t, cell) in cells.enumerated() {
            cell.previousNode = t-1 >= 0 ? cells[t-1] : nil
            cell.nextNode = t+1 < cells.count ? cells[t+1] : nil
        }
    }

    public func initializeForward(builder: ForwardInvocationBuilder, batchSize: Int) throws {
        for cell in cells {
            try cell.initializeForward(builder: builder, batchSize: batchSize)
        }
    }

    public func initializeBackward(builder: BackwardInvocationBuilder, batchSize: Int) throws {
        for cell in cells {
            try cell.initializeBackward(builder: builder, batchSize: batchSize)
        }
    }

    public func encodeParametersUpdate(_ encodeAction: (_ values: Buffer, _ deltas: Buffer) -> Void) {
        for cell in cells {
            cell.encodeParametersUpdate(encodeAction)
        }
    }
}
