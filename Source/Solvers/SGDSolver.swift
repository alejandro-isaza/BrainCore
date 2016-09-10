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

    public var learningRate: Double {
        return learningRateSchedule(initialLearningRate, currentStep)
    }
    public var learningRateSchedule: (Double, Int) -> Double
    public var initialLearningRate: Double

    public internal(set) var currentStep: Int = 0
    public let stepCount: Int
    public var stepAction: ((Snapshot) -> Void)?

    var queue: DispatchQueue
    let trainer: Trainer

    let updateFunctionName = "sgd_update_parameters"
    var updateFunction: MTLComputePipelineState

    public init(net: Net, device: MTLDevice, batchSize: Int, stepCount: Int, initialLearningRate: Double = 0.001, learningRateSchedule: @escaping (Double, Int) -> Double) throws {
        self.net = net
        self.batchSize = batchSize
        self.learningRateSchedule = learningRateSchedule
        self.initialLearningRate = initialLearningRate
        self.stepCount = stepCount

        queue = DispatchQueue(label: "BrainCore.SGDSolver", attributes: [])
        
        guard let trainer = try? Trainer(net: net, device: device, batchSize: batchSize) else {
            fatalError("Could not create trainer.")
        }
        self.trainer = trainer

        let updateParametersLibraryFunction = trainer.library.makeFunction(name: updateFunctionName)!
        updateFunction = try trainer.library.device.makeComputePipelineState(function: updateParametersLibraryFunction)
    }

    public func train(_ completion: @escaping () -> Void) {
        currentStep = 0
        self.queue.async {
            self.step(completion)
        }
    }

    func step(_ completion: @escaping () -> Void) {
        if currentStep >= stepCount {
            completion()
            return
        }

        currentStep += 1

        trainer.run({ snapshot in
            self.queue.async {
                self.completeStep(snapshot: snapshot, completion: completion)
            }
        })
    }

    func completeStep(snapshot: Snapshot, completion: @escaping () -> Void) {
        updateParameters(completion)
        stepAction?(snapshot)
        step(completion)
    }

    func updateParameters(_ completion: () -> Void) {
        let commandBuffer = trainer.commandQueue.makeCommandBuffer()
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
    func encodeUpdateInBuffer(_ buffer: MTLCommandBuffer, values: Buffer, deltas: Buffer) {
        guard let valuesBuffer = values.metalBuffer, let deltasBuffer = deltas.metalBuffer else {
            preconditionFailure("Missing values or deltas buffer for parameter update")
        }

        struct Parameters {
            var learningRate: Float
        }
        var params = Parameters(learningRate: Float(learningRate))
        let paramsBuffer = trainer.device.makeBuffer(bytes: &params, length: MemoryLayout<Parameters>.size, options: .cpuCacheModeWriteCombined)

        var parameterLength = UInt32(valuesBuffer.length / MemoryLayout<Float32>.size)
        let encoder = buffer.makeComputeCommandEncoder()
        encoder.label = "UpdateParameter"
        encoder.setComputePipelineState(updateFunction)
        encoder.setBuffer(valuesBuffer, offset: 0, at: 0)
        encoder.setBuffer(deltasBuffer, offset: 0, at: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, at: 2)
        encoder.setBytes(&parameterLength, length: MemoryLayout<UInt32>.size, at: 3)

        let count = Int(parameterLength)
        let threadsPerGroup = MTLSize(width: updateFunction.threadExecutionWidth, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count - 1) / updateFunction.threadExecutionWidth + 1, height: 1, depth:1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }
}

