// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

/// A data snapshot of a forward-backward pass.
public class Snapshot {
    public let net: Net
    var forwardBuffers: [NSUUID: MTLBuffer]
    var backwardBuffers: [NSUUID: MTLBuffer]

    init(net: Net, forwardBuffers: [NSUUID: MTLBuffer], backwardBuffers: [NSUUID: MTLBuffer] = [NSUUID: MTLBuffer]()) {
        self.net = net
        self.forwardBuffers = forwardBuffers
        self.backwardBuffers = backwardBuffers
    }

    func contentsOfForwardBuffer(buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float>? {
        guard let mtlBuffer = forwardBuffers[buffer.id] else {
            return nil
        }

        return unsafeBufferPointerFromBuffer(mtlBuffer)
    }

    func contentsOfBackwardBuffer(buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float>? {
        guard let mtlBuffer = backwardBuffers[buffer.id] else {
            return nil
        }

        return unsafeBufferPointerFromBuffer(mtlBuffer)
    }

    /// Return a pointer to the forward-pass contents of a network buffer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func contentsOfForwardBuffer(ref: Net.BufferID) -> UnsafeMutableBufferPointer<Float>? {
        guard let buffer = net.buffers[ref] else {
            return nil
        }
        return contentsOfForwardBuffer(buffer)
    }

    /// Return a pointer to the backward-pass contents of a network buffer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func contentsOfBackwardBuffer(ref: Net.BufferID) -> UnsafeMutableBufferPointer<Float>? {
        guard let buffer = net.buffers[ref] else {
            return nil
        }
        return contentsOfBackwardBuffer(buffer)
    }

    /// Return a pointer to the forward-pass output of a layer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func outputOfLayer(layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            bufferId = node.outputBuffer?.id,
            buffer = forwardBuffers[bufferId] else {
                return nil
        }
        let pointer = UnsafeMutablePointer<Float>(buffer.contents())
        let count = buffer.length / sizeof(Float) - node.outputRange.startIndex
        return UnsafeMutableBufferPointer(start: pointer + node.outputRange.startIndex, count: count)
    }

    /// Return a pointer to the forward-pass input of a layer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func inputOfLayer(layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            bufferId = node.inputBuffer?.id,
            buffer = forwardBuffers[bufferId] else {
                return nil
        }
        let pointer = UnsafeMutablePointer<Float>(buffer.contents())
        let count = buffer.length / sizeof(Float) - node.inputRange.startIndex
        return UnsafeMutableBufferPointer(start: pointer + node.inputRange.startIndex, count: count)
    }

    /// Return a pointer to the backward-pass input deltas of a layer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func inputDeltasOfLayer(layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            bufferId = node.inputBuffer?.id,
            buffer = backwardBuffers[bufferId] else {
                return nil
        }
        let pointer = UnsafeMutablePointer<Float>(buffer.contents())
        let count = buffer.length / sizeof(Float) - node.inputRange.startIndex
        return UnsafeMutableBufferPointer(start: pointer + node.inputRange.startIndex, count: count)
    }

    /// Return a pointer to the backward-pass output deltas of a layer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func outputDeltasOfLayer(layer: Layer) -> UnsafeMutableBufferPointer<Float>? {
        guard let node = net.nodeForLayer(layer),
            bufferId = node.outputBuffer?.id,
            buffer = backwardBuffers[bufferId] else {
                return nil
        }
        let pointer = UnsafeMutablePointer<Float>(buffer.contents())
        let count = buffer.length / sizeof(Float) - node.outputRange.startIndex
        return UnsafeMutableBufferPointer(start: pointer + node.outputRange.startIndex, count: count)
    }
}
