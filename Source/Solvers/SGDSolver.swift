// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal


public struct SGDSolverParameters: SolverParameters {
    let learningRate: Float
    let momentum: Float
}

public class SGDSolver: Solver {
    public let updateFunctionName = "sgd_update_parameters"
    
    public let net: Net
    public let batchSize: Int
    
    public internal(set) var currentStep: Int = 0
    public let endStep: Int
    
    public var params: SolverParameters
    public var loss: Double = 0.0
    
    public let runner: Runner

    var queue: dispatch_queue_t
    
    public init(device: MTLDevice, net: Net, batchSize: Int, endStep: Int, learningRate: Double = 0.001, momentum: Double = 0.1) {
        self.net = net
        self.batchSize = batchSize
        self.params = SGDSolverParameters(learningRate: Float(learningRate), momentum: Float(momentum))
        self.endStep = endStep
        
        guard let runner = try? Runner(net: net, device: device, batchSize: batchSize, params: params, updateFunctionName: updateFunctionName) else {
            fatalError("Could not create runner.")
        }
        self.runner = runner

        queue = dispatch_queue_create("BrainCore.Solver", DISPATCH_QUEUE_SERIAL)
    }

    public func train() {
        var done = true

        runner.forwardPassAction = { instance in
            dispatch_async(self.queue) {
                self.runner.backward(instance)
            }
        }

        runner.backwardPassAction = { instance in
            let net = self.runner.net
            let commandBuffer = self.runner.commandQueue.commandBuffer()
            for n in net.nodes {
                if let paramLayer = n.layer as? BackwardParameterLayer {
                    paramLayer.update(commandBuffer, solverParams: instance.solverParametersBuffer)
                }
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            done = true
        }

        while currentStep < endStep {
            if done {
                done = false
                runner.forward()
                currentStep += 1
            }
        }
    }
}

