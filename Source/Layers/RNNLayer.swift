// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class RNNLayer: TrainableLayer, BackwardLayer {
    public let id = NSUUID()
    public let name: String?

    /// The cell copies for each timestep
    public var cells: [LSTMLayer]

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
        return cells.reverse().flatMap({ $0.backwardInvocations })
    }


    public init(cell: LSTMLayer, sequenceLength: Int, name: String? = nil) {
        self.name = name

        self.cells = (0..<sequenceLength).map({ LSTMLayer(weights: cell.weights, biases: cell.biases, continuous: false, time: $0, name: "\(cell.name) T\($0)", clipTo: cell.clipTo) })

        for (t, cell) in cells.enumerate() {
            cell.previousLSTM = t-1 >= 0 ? cells[t-1] : nil
            cell.nextLSTM = t+1 < cells.count ? cells[t+1] : nil
        }
    }

    public func initializeForward(builder builder: ForwardInvocationBuilder, batchSize: Int) throws {
        for cell in cells {
            try cell.initializeForward(builder: builder, batchSize: batchSize)
        }
    }

    public func initializeBackward(builder builder: BackwardInvocationBuilder, batchSize: Int) throws {
        for cell in cells {
            try cell.initializeBackward(builder: builder, batchSize: batchSize)
        }
    }

    public func encodeParametersUpdate(encodeAction: (values: Buffer, deltas: Buffer) -> Void) {
        for cell in cells {
            cell.encodeParametersUpdate(encodeAction)
        }
    }
}
