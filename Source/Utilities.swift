// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

/// Convert a Metal buffer to an `UnsafeMutableBufferPointer`
public func unsafeBufferPointerFromBuffer(buffer: MTLBuffer) -> UnsafeMutableBufferPointer<Float> {
    let pointer = UnsafeMutablePointer<Float>(buffer.contents())
    let count = buffer.length / sizeof(Float)
    return UnsafeMutableBufferPointer(start: pointer, count: count)
}

/// Copy the contents of a Metal buffer to an array
public func arrayFromBuffer(buffer: MTLBuffer, start: Int = 0, count: Int? = nil) -> Array<Float> {
    let pointer = UnsafePointer<Float>(buffer.contents()) + start
    let count = count ?? buffer.length / sizeof(Float) - start
    return Array<Float>(UnsafeBufferPointer(start: pointer, count: count))
}

/// Copy the contents of a Metal buffer to a ValueArray
public func valueArrayFromBuffer(buffer: MTLBuffer, start: Int = 0, count: Int? = nil) -> ValueArray<Float> {
    let pointer = UnsafeMutablePointer<Float>(buffer.contents()) + start
    let count = count ?? buffer.length / sizeof(Float) - start
    var array = ValueArray<Float>(count: count)
    withPointer(&array) { $0.assignFrom(pointer, count: count) }
    return array
}

/// Copy a collection of values to a Metal buffer
public func fillBuffer<Collection: CollectionType where Collection.Generator.Element == Float>(buffer: MTLBuffer, start: Int, withElements elements: Collection) {
    let pointer = UnsafeMutablePointer<Float>(buffer.contents()) + start
    for (i, v) in elements.enumerate() {
        pointer[i] = v
    }
}

/// Create a Metal buffer from a tensor
public func createBuffer<T: TensorType where T.Element == Float>(inDevice device: MTLDevice, fromTensor tensor: T, withLabel label: String) -> MTLBuffer {
    return withPointer(tensor) { pointer in
        let tempBuffer = device.newBufferWithBytes(pointer, length: tensor.count * sizeof(Float), options: .CPUCacheModeDefaultCache)
        precondition(tempBuffer.length == tensor.count * sizeof(Float), "Failed to allocate \(tensor.count * sizeof(Float))B")
        tempBuffer.label = label

        return tempBuffer
    }
}

/// Create a Metal buffer from a pointer
public func createBuffer(inDevice device: MTLDevice, fromPointer pointer: UnsafePointer<Void>, ofSize size: Int, withLabel label: String) -> MTLBuffer {
    let buffer = device.newBufferWithBytes(pointer, length: size, options: .CPUCacheModeDefaultCache)
    precondition(buffer.length == size, "Failed to allocate \(size)B")
    buffer.label = label

    return buffer
}

/// Create an empty Metal buffer with specific size
public func createBuffer(inDevice device: MTLDevice, ofSize size: Int, withLabel label: String) -> MTLBuffer {
    let buffer = device.newBufferWithLength(size, options: .CPUCacheModeDefaultCache)
    precondition(buffer.length == size, "Failed to allocate \(size)B")
    buffer.label = label

    return buffer
}
