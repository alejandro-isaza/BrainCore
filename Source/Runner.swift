// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Foundation
import Metal

public class Runner {
    public let net: Net
    public let batchSize: Int

    let device: MTLDevice
    var library: MTLLibrary!
    var commandQueue: MTLCommandQueue

    public init(net: Net, device: MTLDevice, batchSize: Int) throws {
        self.net = net
        self.net.insertTransposeLayers()
        self.batchSize = batchSize

        self.device = device

        commandQueue = device.newCommandQueue()
        commandQueue.label = "BrainCore.Runner"

        guard let path = NSBundle(forClass: self.dynamicType).pathForResource("default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        library = try device.newLibraryWithFile(path)

        for node in net.nodes.values {
            if let forwardLayer = node.layer as? ForwardLayer {
                try initializeForwardNode(node, layer: forwardLayer)
            } else if let backwardLayer = node.layer as? BackwardLayer {
                try initializeBackwardNode(node, layer: backwardLayer)
            } else if let lossLayer = node.layer as? LossLayer {
                try initializeLossNode(node, layer: lossLayer)
            }
        }
    }

    func initializeForwardNode(node: NetNode, layer: ForwardLayer) throws {
        guard let inputNetBuffer = node.inputBuffer else {
            fatalError("Input buffer for \(layer.name) not found.")
        }

        let inputBuffer = Buffer(
            name: inputNetBuffer.name ?? "input",
            size: 0,
            netBuffer: inputNetBuffer,
            offset: node.inputOffset)

        guard let outputNetBuffer = node.outputBuffer else {
            fatalError("Output buffer for \(layer.name) not found.")
        }
        let outputBuffer = Buffer(
            name: outputNetBuffer.name ?? "output",
            size: 0,
            netBuffer: outputNetBuffer,
            offset: node.outputOffset)

        let invocationBuilder = ForwardInvocationBuilder(device: device, library: library, inputBuffer: inputBuffer, outputBuffer: outputBuffer)
        try layer.initializeForward(builder: invocationBuilder, batchSize: 1)
    }

    func initializeBackwardNode(node: NetNode, layer: BackwardLayer) throws {
        guard let inputNetBuffer = node.inputBuffer else {
            fatalError("Input buffer for \(layer.name) not found.")
        }

        let inputBuffer = Buffer(
            name: inputNetBuffer.name ?? "input",
            size: 0,
            netBuffer: inputNetBuffer,
            offset: node.inputOffset)

        let inputDeltasNetBuffer = NetBuffer(id: inputNetBuffer.id, type: .Deltas, name: inputNetBuffer.name)
        let inputDeltasBuffer = Buffer(
            name: inputNetBuffer.name ?? "input deltas",
            size: 0,
            netBuffer: inputDeltasNetBuffer,
            offset: node.inputOffset)

        guard let outputNetBuffer = node.outputBuffer else {
            fatalError("Output buffer for \(layer.name) not found.")
        }
        let outputDeltasNetBuffer = NetBuffer(id: outputNetBuffer.id, type: .Deltas, name: outputNetBuffer.name)
        let outputDeltasBuffer = Buffer(
            name: outputNetBuffer.name ?? "output deltas",
            size: 0,
            netBuffer: outputDeltasNetBuffer,
            offset: node.outputOffset)

        let invocationBuilder = BackwardInvocationBuilder(device: device, library: library, inputBuffer: inputBuffer, outputDeltasBuffer: outputDeltasBuffer, inputDeltasBuffer: inputDeltasBuffer)
        try layer.initializeBackward(builder: invocationBuilder, batchSize: 1)
    }

    func initializeLossNode(node: NetNode, layer: LossLayer) throws {
        guard let inputNetBuffer = node.inputBuffer else {
            fatalError("Input buffer for \(layer.name) not found.")
        }

        let inputBuffer = Buffer(
            name: inputNetBuffer.name ?? "input",
            size: 0,
            netBuffer: inputNetBuffer,
            offset: node.inputOffset)

        let inputDeltasNetBuffer = NetBuffer(id: inputNetBuffer.id, type: .Deltas, name: inputNetBuffer.name)
        let inputDeltasBuffer = Buffer(
            name: inputNetBuffer.name ?? "input deltas",
            size: 0,
            netBuffer: inputDeltasNetBuffer,
            offset: node.inputOffset)

        guard let outputNetBuffer = node.outputBuffer else {
            fatalError("Output buffer for \(layer.name) not found.")
        }
        let outputDeltasNetBuffer = NetBuffer(id: outputNetBuffer.id, type: .Deltas, name: outputNetBuffer.name)
        let outputDeltasBuffer = Buffer(
            name: outputNetBuffer.name ?? "output deltas",
            size: 0,
            netBuffer: outputDeltasNetBuffer,
            offset: node.outputOffset)

        let invocationBuilder = BackwardInvocationBuilder(device: device, library: library, inputBuffer: inputBuffer, outputDeltasBuffer: outputDeltasBuffer, inputDeltasBuffer: inputDeltasBuffer)
        try layer.initializeBackward(builder: invocationBuilder, batchSize: 1)
    }

    /// Encode an invocation
    func encode(invocation invocation: Invocation, forNode node: NetNode, commandBuffer: MTLCommandBuffer) throws {
        let encoder = commandBuffer.computeCommandEncoder()
        encoder.setComputePipelineState(invocation.pipelineState)

        for (index, buffer) in invocation.buffers.enumerate() {
            encoder.setBuffer(buffer.metalBuffer!, offset: buffer.metalBufferOffset * batchSize, atIndex: index)
        }

        for (index, value) in invocation.values.enumerate() {
            var mutableValue = value
            encoder.setBytes(&mutableValue, length: sizeofValue(value), atIndex: index + invocation.buffers.count)
        }

        let threadWidth = invocation.pipelineState.threadExecutionWidth
        let threadsPerGroup = MTLSize(width: threadWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (invocation.width - 1) / threadWidth + 1, height: invocation.height, depth: invocation.depth)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}
