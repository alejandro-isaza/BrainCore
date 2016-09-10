// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

/// A data snapshot of a forward-backward pass.
open class Snapshot {
    /// The network definition.
    open let net: Net

    var forwardBuffers: [UUID: MTLBuffer]
    var backwardBuffers: [UUID: MTLBuffer]

    init(net: Net, forwardBuffers: [UUID: MTLBuffer], backwardBuffers: [UUID: MTLBuffer] = [UUID: MTLBuffer]()) {
        self.net = net
        self.forwardBuffers = forwardBuffers
        self.backwardBuffers = backwardBuffers
    }

    func contentsOfForwardBuffer(_ buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float>? {
        guard let mtlBuffer = forwardBuffers[buffer.id as UUID] else {
            return nil
        }

        return unsafeBufferPointerFromBuffer(mtlBuffer)
    }

    func contentsOfBackwardBuffer(_ buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float>? {
        guard let mtlBuffer = backwardBuffers[buffer.id as UUID] else {
            return nil
        }

        return unsafeBufferPointerFromBuffer(mtlBuffer)
    }

    /// Returns a pointer to the forward-pass contents of a network buffer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func contentsOfForwardBuffer(_ ref: Net.BufferID) -> UnsafeMutableBufferPointer<Float>? {
        guard let buffer = net.buffers[ref] else {
            return nil
        }
        return contentsOfForwardBuffer(buffer)
    }

    /// Returns a pointer to the backward-pass contents of a network buffer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func contentsOfBackwardBuffer(_ ref: Net.BufferID) -> UnsafeMutableBufferPointer<Float>? {
        guard let buffer = net.buffers[ref] else {
            return nil
        }
        return contentsOfBackwardBuffer(buffer)
    }

    /// Returns a pointer to the forward-pass output of a layer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func outputOfLayer(_ layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            let bufferId = node.outputBuffer?.id,
            let buffer = forwardBuffers[bufferId] else {
                return nil
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size)
        let count = buffer.length / MemoryLayout<Float>.size - node.outputRange.lowerBound
        return UnsafeMutableBufferPointer(start: pointer + node.outputRange.lowerBound, count: count)
    }

    /// Returns a pointer to the forward-pass input of a layer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func inputOfLayer(_ layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            let bufferId = node.inputBuffer?.id,
            let buffer = forwardBuffers[bufferId] else {
                return nil
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size)
        let count = buffer.length / MemoryLayout<Float>.size - node.inputRange.lowerBound
        return UnsafeMutableBufferPointer(start: pointer + node.inputRange.lowerBound, count: count)
    }

    /// Returns a pointer to the backward-pass input deltas of a layer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func inputDeltasOfLayer(_ layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            let bufferId = node.inputBuffer?.id,
            let buffer = backwardBuffers[bufferId] else {
                return nil
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size)
        let count = buffer.length / MemoryLayout<Float>.size - node.inputRange.lowerBound
        return UnsafeMutableBufferPointer(start: pointer + node.inputRange.lowerBound, count: count)
    }

    /// Returns a pointer to the backward-pass output deltas of a layer.
    ///
    /// - Note: The pointer is short-lived, you should copy any contents that you want preserve.
    open func outputDeltasOfLayer(_ layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            let bufferId = node.outputBuffer?.id,
            let buffer = backwardBuffers[bufferId] else {
                return nil
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: buffer.length / MemoryLayout<Float>.size)
        let count = buffer.length / MemoryLayout<Float>.size - node.outputRange.lowerBound
        return UnsafeMutableBufferPointer(start: pointer + node.outputRange.lowerBound, count: count)
    }
}
