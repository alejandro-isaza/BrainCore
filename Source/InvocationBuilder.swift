// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

public class InvocationBuilder {
    let device: MTLDevice
    let library: MTLLibrary

    init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    /// Create a buffer initialized with the given elements
    public func createBuffer<T: TensorType where T.Element == Float>(name name: String, elements: T) -> Buffer {
        let size = elements.count * sizeof(Float)
        let buffer = withPointer(elements) { (elementsPointer: UnsafePointer<Float>) -> MTLBuffer in
            let length = size + sizeof(GPUBufferHeader)
            let buffer = device.newBufferWithLength(length, options: .CPUCacheModeDefaultCache)

            let header = GPUBufferHeader(inputSize: size, sequenceSize: 1, batchSize: 1)
            UnsafeMutablePointer<GPUBufferHeader>(buffer.contents()).memory = header

            let dataPointer = UnsafeMutablePointer<Float>(buffer.contents() + sizeof(GPUBufferHeader))
            dataPointer.assignFrom(UnsafeMutablePointer(elementsPointer), count: size)

            return buffer
        }
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Create an uninitialized buffer
    public func createBuffer(name name: String, size: Int) -> Buffer {
        let length = size + sizeof(GPUBufferHeader)
        let buffer = device.newBufferWithLength(length, options: .CPUCacheModeDefaultCache)
        precondition(buffer.length == size, "Failed to allocate \(size)B")
        buffer.label = name

        let header = GPUBufferHeader(inputSize: size, sequenceSize: 1, batchSize: 1)
        UnsafeMutablePointer<GPUBufferHeader>(buffer.contents()).memory = header

        return Buffer(name: name, size: size, metalBuffer: buffer, offset: 0)
    }

    /// Create an invocation
    public func createInvocation(functionName functionName: String, buffers: [Buffer], values: [Any], width: Int = 1, height: Int = 1, depth: Int = 1) throws -> Invocation {
        guard let function = library.newFunctionWithName(functionName) else {
            fatalError("Metal function not found '\(functionName)'")
        }

        let pipelineState = try library.device.newComputePipelineStateWithFunction(function)
        return Invocation(functionName: functionName, buffers: buffers, values: values, width: width, height: height, depth: depth, pipelineState: pipelineState)
    }
}

public class ForwardInvocationBuilder: InvocationBuilder {
    public internal(set) var inputBuffer: Buffer
    public internal(set) var outputBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        super.init(device: device, library: library)
    }
}

public class BackwardInvocationBuilder: InvocationBuilder {
    public internal(set) var inputBuffer: Buffer
    public internal(set) var outputDeltasBuffer: Buffer
    public internal(set) var inputDeltasBuffer: Buffer

    init(device: MTLDevice, library: MTLLibrary, inputBuffer: Buffer, outputDeltasBuffer: Buffer, inputDeltasBuffer: Buffer) {
        self.inputBuffer = inputBuffer
        self.outputDeltasBuffer = outputDeltasBuffer
        self.inputDeltasBuffer = inputDeltasBuffer
        super.init(device: device, library: library)
    }
}
