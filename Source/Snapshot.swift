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
    var forwardBuffers: [MTLBuffer]
    var backwardBuffers: [MTLBuffer]

    init(net: Net, forwardBuffers: [MTLBuffer], backwardBuffers: [MTLBuffer] = []) {
        self.net = net
        self.forwardBuffers = forwardBuffers
        self.backwardBuffers = backwardBuffers
    }

    func forwardContentsOfBuffer(buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float> {
        return unsafeBufferPointerFromBuffer(forwardBuffers[buffer.id])
    }

    func backwardContentsOfBuffer(buffer: NetBuffer) -> UnsafeMutableBufferPointer<Float> {
        return unsafeBufferPointerFromBuffer(backwardBuffers[buffer.id])
    }

    /// Return a pointer to the forward-pass contents of a network buffer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func forwardContentsOfBufferNamed(name: String) -> UnsafeMutableBufferPointer<Float>? {
        for buffer in net.buffers {
            if buffer.name == name {
                return forwardContentsOfBuffer(buffer)
            }
        }
        return nil
    }

    /// Return a pointer to the backward-pass contents of a network buffer. The pointer is short-lived, you should copy any contents that you want preserve.
    public func backwardContentsOfBufferNamed(name: String) -> UnsafeMutableBufferPointer<Float>? {
        for buffer in net.buffers {
            if buffer.name == name {
                return backwardContentsOfBuffer(buffer)
            }
        }
        return nil
    }

}
