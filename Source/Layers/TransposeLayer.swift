// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

/// TransposeLayer transposes input data so that elements in consecutive batches are contiguous in memory. We do this so that concatenation of layer outputs becomes concatenation of memory blocks removing the need of concat and split layers. This class does not perform matrix transposition in the general sense and therefore is an internal class to avoid confusion.
internal class TransposeLayer: ForwardLayer {
    struct Parameters {
        let batchSize: UInt32
        let inputSize: UInt32
    }

    let id = UUID()
    let name: String?

    /// The size of each batch element
    let size: Int

    var outputSize: Int {
        return size
    }

    var inputSize: Int {
        return size
    }

    var forwardInvocation: Invocation?

    var forwardInvocations: [Invocation] {
        guard let forwardInvocation = forwardInvocation else {
            fatalError("initializeForward needs to be called first")
        }
        return [forwardInvocation]
    }

    init(size: Int, name: String? = nil) {
        self.name = name
        self.size = size
    }

    func initializeForward(builder: ForwardInvocationBuilder, batchSize: Int) throws {
        let buffers = [
            builder.inputBuffer,
            builder.outputBuffer
        ]

        let params = Parameters(batchSize: UInt32(batchSize), inputSize: UInt32(inputSize))
        forwardInvocation = try builder.createInvocation(
            functionName: "transpose",
            buffers: buffers,
            values: [params],
            width: size,
            height: batchSize
        )
    }
}
