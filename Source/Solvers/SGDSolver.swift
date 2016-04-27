// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public class SGDSolver {
    public let net: Net
    public let batchSize: Int
    public var learningRate: Double
    public var momentum: Double
    
    public internal(set) var currentStep: Int = 0
    public let stepCount: Int
    public var stepAction: ((Snapshot) -> Void)?

    var queue: dispatch_queue_t
    let trainer: Trainer

    let updateFunctionName = "sgd_update_parameters"
    var updateFunction: MTLComputePipelineState

    public init(net: Net, device: MTLDevice, batchSize: Int, stepCount: Int, learningRate: Double = 0.001, momentum: Double = 0.1) throws {
        self.net = net
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.momentum = momentum
        self.stepCount = stepCount

        queue = dispatch_queue_create("BrainCore.SGDSolver", DISPATCH_QUEUE_SERIAL)
        
        guard let trainer = try? Trainer(net: net, device: device, batchSize: batchSize) else {
            fatalError("Could not create trainer.")
        }
        self.trainer = trainer

        let updateParametersLibraryFunction = trainer.library.newFunctionWithName(updateFunctionName)!
        updateFunction = try trainer.library.device.newComputePipelineStateWithFunction(updateParametersLibraryFunction)
    }

    public func train(completion: () -> Void) {
        currentStep = 0
        dispatch_async(self.queue) {
            self.step(completion)
        }
    }

    func step(completion: () -> Void) {
        if currentStep >= stepCount {
            completion()
            return
        }

        learningRate = learningRate / (1 + momentum * Double(currentStep))
        currentStep += 1

        trainer.run({ snapshot in
            dispatch_async(self.queue) {
                self.completeStep(snapshot: snapshot, completion: completion)
            }
        })
    }

    func completeStep(snapshot snapshot: Snapshot, completion: () -> Void) {
        updateParameters(completion)
        stepAction?(snapshot)
        step(completion)
    }

    func updateParameters(completion: () -> Void) {
        let commandBuffer = trainer.commandQueue.commandBuffer()
        for node in trainer.net.nodes.values {
            guard let trainableLayer = node.layer as? TrainableLayer else {
                continue
            }
            trainableLayer.encodeParametersUpdate({ values, deltas in
                self.encodeUpdateInBuffer(commandBuffer, values: values, deltas: deltas)
            })
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    /// Performs a parameter update on the GPU.
    func encodeUpdateInBuffer(buffer: MTLCommandBuffer, values: Buffer, deltas: Buffer) {
        guard let valuesBuffer = values.metalBuffer, deltasBuffer = deltas.metalBuffer else {
            preconditionFailure("Missing values or deltas buffer for parameter update")
        }

        struct Parameters {
            var learningRate: Float
            var momentum: Float
        }
        var params = Parameters(learningRate: Float(learningRate), momentum: Float(momentum))
        let paramsBuffer = trainer.device.newBufferWithBytes(&params, length: sizeof(Parameters), options: .CPUCacheModeWriteCombined)

        var parameterLength = UInt32(valuesBuffer.length / sizeof(Float32))
        let encoder = buffer.computeCommandEncoder()
        encoder.label = "UpdateParameter"
        encoder.setComputePipelineState(updateFunction)
        encoder.setBuffer(valuesBuffer, offset: 0, atIndex: 0)
        encoder.setBuffer(deltasBuffer, offset: 0, atIndex: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, atIndex: 2)
        encoder.setBytes(&parameterLength, length: sizeof(parameterLength.dynamicType), atIndex: 3)

        let count = Int(parameterLength)
        let threadsPerGroup = MTLSize(width: updateFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / updateFunction.threadExecutionWidth + 1, height: 1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}

