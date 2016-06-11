// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

/// A reference to a GPU buffer used by an `Invocation`.
public class Buffer {
    /// The name of the buffer
    public let name: String

    /// The size of the buffer contents in bytes
    public let size: Int

    var netBuffer: NetBuffer?
    public var metalBuffer: MTLBuffer?
    public var metalBufferOffset = 0

    init(name: String, size: Int, netBuffer: NetBuffer, offset: Int = 0) {
        self.name = name
        self.size = size
        self.netBuffer = netBuffer
        self.metalBufferOffset = offset
    }

    init(name: String, size: Int, metalBuffer: MTLBuffer, offset: Int = 0) {
        self.name = name
        self.size = size
        self.metalBuffer = metalBuffer
        self.metalBufferOffset = offset
    }
}

/// A GPU function invocation.
public class Invocation {
    /// The function name.
    public let functionName: String

    /// The array of buffer arguments. Used for data that is reused across executions.
    public var buffers: [Buffer]

    /// The array of non-buffer arguments. Used for data that is not reused across executions.
    public var values: [Any]

    /// The execution grid width.
    public var width = 1

    /// The execution grid height.
    public var height = 1

    /// The execution grid depth.
    public var depth = 1

    let pipelineState: MTLComputePipelineState

    init(functionName: String, buffers: [Buffer], values: [Any], width: Int = 1, height: Int = 1, depth: Int = 1, pipelineState: MTLComputePipelineState) {
        self.functionName = functionName
        self.buffers = buffers
        self.values = values
        self.width = width
        self.height = height
        self.depth = depth
        self.pipelineState = pipelineState
    }
}
