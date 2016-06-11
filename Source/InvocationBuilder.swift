// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

/// Utility class to create invocations.
public class InvocationBuilder {
    let device: MTLDevice
    let library: MTLLibrary

    init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    /// Creates a buffer initialized with the given elements.
    public func createBuffer<T: TensorType where T.Element == Float>(name name: String, elements: T) -> Buffer {
        let size = elements.count * sizeof(Float)
        let buffer = withPointer(elements) { elementsPointer in
            return device.newBufferWithBytes(elementsPointer, length: size, options: .CPUCacheModeDefaultCache)
        }
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Creates an uninitialized buffer.
    public func createBuffer(name name: String, size: Int) -> Buffer {
        let buffer = device.newBufferWithLength(size, options: .CPUCacheModeDefaultCache)
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Creates an invocation.
    public func createInvocation(functionName functionName: String, buffers: [Buffer], values: [Any], width: Int = 1, height: Int = 1, depth: Int = 1) throws -> Invocation {
        guard let function = library.newFunctionWithName(functionName) else {
            fatalError("Metal function not found '\(functionName)'")
        }

        let pipelineState = try library.device.newComputePipelineStateWithFunction(function)
        return Invocation(functionName: functionName, buffers: buffers, values: values, width: width, height: height, depth: depth, pipelineState: pipelineState)
    }
}

/// An `InvocationBuilder` for feed-forward invocations.
public class ForwardInvocationBuilder: InvocationBuilder {
    /// The `Buffer` containing the layer's input data.
    public internal(set) var inputBuffer: Buffer

    /// The `Buffer` for the layer's output data.
    public internal(set) var outputBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        super.init(device: device, library: library)
    }
}

/// An `InvocationBuilder` for backpropagation invocations.
public class BackwardInvocationBuilder: InvocationBuilder {
    /// The `Buffer` containing the layer's input data.
    public internal(set) var inputBuffer: Buffer

    /// The `Buffer` containing the layer's output deltas.
    public internal(set) var outputDeltasBuffer: Buffer

    /// The `Buffer` for the layer's input deltas.
    public internal(set) var inputDeltasBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputDeltasBuffer: Buffer, inputDeltasBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputDeltasBuffer = outputDeltasBuffer
        self.inputDeltasBuffer = inputDeltasBuffer
        super.init(device: device, library: library)
    }
}
