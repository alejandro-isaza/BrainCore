// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

/// A base class that sets up a network definition to be exectued, either feed-forward or backpropagation, on a GPU.
open class Runner {
    /// The network definition.
    open let net: Net

    /// The batch size.
    open let batchSize: Int

    /// The Metal GPU device to use.
    let device: MTLDevice

    /// The Metal library with the layers' GPU functions.
    var library: MTLLibrary!

    /// The Metal command queue.
    var commandQueue: MTLCommandQueue

    /// Creates a `Runner` for the given network definition.
    ///
    /// - Parameter net:       network definition.
    /// - Parameter device:    Metal device to use when running.
    /// - Parameter batchSize: batch size.
    /// - Parameter backwards: determines if the `Runner` will support running backpropagation.
    public init(net: Net, device: MTLDevice, batchSize: Int, backwards: Bool) throws {
        self.net = net
        self.net.insertTransposeLayers()
        self.batchSize = batchSize

        self.device = device

        commandQueue = device.makeCommandQueue()
        commandQueue.label = "BrainCore.Runner"

        guard let path = Bundle(for: type(of: self)).path(forResource: "default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        library = try device.makeLibrary(filepath: path)

        for node in net.nodes.values {
            if let forwardLayer = node.layer as? ForwardLayer {
                try initializeForwardNode(node, layer: forwardLayer)
            }
            if backwards {
                if let backwardLayer = node.layer as? BackwardLayer {
                    try initializeBackwardNode(node, layer: backwardLayer)
                }
            }
        }
    }

    /// Initializes a network node for feed-forward execution.
    func initializeForwardNode(_ node: NetNode, layer: ForwardLayer) throws {
        guard let inputNetBuffer = node.inputBuffer else {
            fatalError("Input buffer for \(layer.name) not found.")
        }

        let inputBuffer = Buffer(
            name: inputNetBuffer.name ?? "input",
            size: inputNetBuffer.size,
            netBuffer: inputNetBuffer,
            offset: batchSize * node.inputRange.lowerBound * MemoryLayout<Float>.size)

        guard let outputNetBuffer = node.outputBuffer else {
            fatalError("Output buffer for \(layer.name) not found.")
        }
        let outputBuffer = Buffer(
            name: outputNetBuffer.name ?? "output",
            size: outputNetBuffer.size,
            netBuffer: outputNetBuffer,
            offset: batchSize * node.outputRange.lowerBound * MemoryLayout<Float>.size)

        let invocationBuilder = ForwardInvocationBuilder(device: device, library: library, inputBuffer: inputBuffer, outputBuffer: outputBuffer)
        try layer.initializeForward(builder: invocationBuilder, batchSize: batchSize)
    }

    /// Initializes a network node for backpropagation.
    func initializeBackwardNode(_ node: NetNode, layer: BackwardLayer) throws {
        guard let inputNetBuffer = node.inputBuffer else {
            fatalError("Input buffer for \(layer.name) not found.")
        }

        let inputBuffer = Buffer(
            name: inputNetBuffer.name ?? "input",
            size: inputNetBuffer.size,
            netBuffer: inputNetBuffer,
            offset: batchSize * node.inputRange.lowerBound * MemoryLayout<Float>.size)

        let inputDeltasNetBuffer = NetBuffer(id: inputNetBuffer.id, type: .deltas, name: inputNetBuffer.name)
        let inputDeltasBuffer = Buffer(
            name: inputDeltasNetBuffer.name ?? "input deltas",
            size: inputDeltasNetBuffer.size,
            netBuffer: inputDeltasNetBuffer,
            offset: batchSize * node.inputRange.lowerBound * MemoryLayout<Float>.size)

        guard let outputNetBuffer = node.outputBuffer else {
            fatalError("Output buffer for \(layer.name) not found.")
        }
        let outputDeltasNetBuffer = NetBuffer(id: outputNetBuffer.id, type: .deltas, name: outputNetBuffer.name)
        let outputDeltasBuffer = Buffer(
            name: outputDeltasNetBuffer.name ?? "output deltas",
            size: outputDeltasNetBuffer.size,
            netBuffer: outputDeltasNetBuffer,
            offset: batchSize * node.outputRange.lowerBound * MemoryLayout<Float>.size)

        let invocationBuilder = BackwardInvocationBuilder(device: device, library: library, inputBuffer: inputBuffer, outputDeltasBuffer: outputDeltasBuffer, inputDeltasBuffer: inputDeltasBuffer)
        try layer.initializeBackward(builder: invocationBuilder, batchSize: batchSize)
    }

    /// Encodes an invocation into a command buffer.
    open static func encode(invocation: Invocation, commandBuffer: MTLCommandBuffer) throws {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(invocation.pipelineState)

        for (index, buffer) in invocation.buffers.enumerated() {
            encoder.setBuffer(buffer.metalBuffer!, offset: buffer.metalBufferOffset, at: index)
        }

        for (index, value) in invocation.values.enumerated() {
            var mutableValue = value
            encoder.setBytes(&mutableValue, length: MemoryLayout.size(ofValue: value), at: index + invocation.buffers.count)
        }

        let threadWidth = invocation.pipelineState.threadExecutionWidth
        let threadsPerGroup = MTLSize(width: threadWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (invocation.width - 1) / threadWidth + 1, height: invocation.height, depth: invocation.depth)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}
