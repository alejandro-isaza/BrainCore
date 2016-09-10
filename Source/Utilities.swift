// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal
import Upsurge

/// Converts a Metal buffer to an `UnsafeMutableBufferPointer`.
public func unsafeBufferPointerFromBuffer(_ buffer: MTLBuffer) -> UnsafeMutableBufferPointer<Float> {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size)
    let count = buffer.length / MemoryLayout<Float>.size
    return UnsafeMutableBufferPointer(start: pointer, count: count)
}

/// Copies the contents of a Metal buffer to an array.
public func arrayFromBuffer(_ buffer: MTLBuffer, start: Int = 0, count: Int? = nil) -> Array<Float> {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size).advanced(by: start)
    let count = count ?? buffer.length / MemoryLayout<Float>.size - start
    return Array<Float>(UnsafeBufferPointer(start: pointer, count: count))
}

/// Copies the contents of a Metal buffer to a ValueArray.
public func valueArrayFromBuffer(_ buffer: MTLBuffer, start: Int = 0, count: Int? = nil) -> ValueArray<Float> {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size).advanced(by: start)
    let count = count ?? buffer.length / MemoryLayout<Float>.size - start
    var array = ValueArray<Float>(count: count)
    withPointer(&array) { $0.assign(from: pointer, count: count) }
    return array
}

/// Copies a collection of values to a Metal buffer.
public func fillBuffer<Collection: Swift.Collection>(_ buffer: MTLBuffer, start: Int, withElements elements: Collection) where Collection.Iterator.Element == Float {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size).advanced(by: start)
    for (i, v) in elements.enumerated() {
        pointer[i] = v
    }
}

/// Creates a Metal buffer from a tensor.
public func createBuffer<T: TensorType>(inDevice device: MTLDevice, fromTensor tensor: T, withLabel label: String) -> MTLBuffer where T.Element == Float {
    return withPointer(tensor) { pointer in
        let tempBuffer = device.makeBuffer(bytes: pointer, length: tensor.count * MemoryLayout<Float>.size, options: [])
        precondition(tempBuffer.length == tensor.count * MemoryLayout<Float>.size, "Failed to allocate \(tensor.count * MemoryLayout<Float>.size)B")
        tempBuffer.label = label

        return tempBuffer
    }
}

/// Creates a Metal buffer from a pointer.
public func createBuffer(inDevice device: MTLDevice, fromPointer pointer: UnsafeRawPointer, ofSize size: Int, withLabel label: String) -> MTLBuffer {
    let buffer = device.makeBuffer(bytes: pointer, length: size, options: MTLResourceOptions())
    precondition(buffer.length == size, "Failed to allocate \(size)B")
    buffer.label = label

    return buffer
}

/// Creates an empty Metal buffer with a specific size.
public func createBuffer(inDevice device: MTLDevice, ofSize size: Int, withLabel label: String) -> MTLBuffer {
    let buffer = device.makeBuffer(length: size, options: MTLResourceOptions())
    precondition(buffer.length == size, "Failed to allocate \(size)B")
    buffer.label = label

    return buffer
}
