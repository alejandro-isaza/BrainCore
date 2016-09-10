// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

/// Utility class to create invocations.
open class InvocationBuilder {
    let device: MTLDevice
    let library: MTLLibrary

    init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    /// Creates a buffer initialized with the given elements.
    open func createBuffer<T: TensorType>(name: String, elements: T) -> Buffer where T.Element == Float {
        let size = elements.count * MemoryLayout<Float>.size
        let buffer = withPointer(elements) { elementsPointer in
            return device.makeBuffer(bytes: elementsPointer, length: size, options: [])
        }
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Creates an uninitialized buffer.
    open func createBuffer(name: String, size: Int) -> Buffer {
        let buffer = device.makeBuffer(length: size, options: MTLResourceOptions())
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Creates an invocation.
    open func createInvocation(functionName: String, buffers: [Buffer], values: [Any], width: Int = 1, height: Int = 1, depth: Int = 1) throws -> Invocation {
        guard let function = library.makeFunction(name: functionName) else {
            fatalError("Metal function not found '\(functionName)'")
        }

        let pipelineState = try library.device.makeComputePipelineState(function: function)
        return Invocation(functionName: functionName, buffers: buffers, values: values, width: width, height: height, depth: depth, pipelineState: pipelineState)
    }
}

/// An `InvocationBuilder` for feed-forward invocations.
open class ForwardInvocationBuilder: InvocationBuilder {
    /// The `Buffer` containing the layer's input data.
    open internal(set) var inputBuffer: Buffer

    /// The `Buffer` for the layer's output data.
    open internal(set) var outputBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        super.init(device: device, library: library)
    }
}

/// An `InvocationBuilder` for backpropagation invocations.
open class BackwardInvocationBuilder: InvocationBuilder {
    /// The `Buffer` containing the layer's input data.
    open internal(set) var inputBuffer: Buffer

    /// The `Buffer` containing the layer's output deltas.
    open internal(set) var outputDeltasBuffer: Buffer

    /// The `Buffer` for the layer's input deltas.
    open internal(set) var inputDeltasBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputDeltasBuffer: Buffer, inputDeltasBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputDeltasBuffer = outputDeltasBuffer
        self.inputDeltasBuffer = inputDeltasBuffer
        super.init(device: device, library: library)
    }
}
