// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Accelerate
@testable import BrainCore
import Upsurge

class SGDSolverTests: MetalTestCase {
    func testSimpleTrain() {
        let inputSize = 2
        let ip1OutputSize = 2
        let outputSize = 2
        let labelSize = 2
        let batchSize = 1

        let ip1Weights = Matrix<Float>(rows: inputSize, columns: ip1OutputSize, elements: [0.15, 0.25, 0.20, 0.30])
        let ip1Biases: ValueArray<Float> = [0.35, 0.35]

        let ip2Weights = Matrix<Float>(rows: ip1OutputSize, columns: outputSize, elements: [0.40, 0.50, 0.45, 0.55])
        let ip2Biases: ValueArray<Float> = [0.60, 0.60]

        let inputs: ValueArray<Float> = [0.05, 0.10]
        let labels: ValueArray<Float> = [0.01, 0.99]

        let data = Source(name: "data", data: inputs, batchSize: batchSize)
        let label = Source(name: "label", data: labels, batchSize: batchSize)
        let ip1 = InnerProductLayer(weights: ip1Weights, biases: ip1Biases, name: "ip1")
        let sigmoid1 = SigmoidLayer(size: outputSize, name: "sigmoid1")
        let ip2 = InnerProductLayer(weights: ip2Weights, biases: ip2Biases, name: "ip2")
        let sigmoid2 = SigmoidLayer(size: outputSize, name: "sigmoid2")
        let loss = L2LossLayer(size: labelSize, name: "loss")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: batchSize)

        let net = Net.build({
            data => ip1 => sigmoid1 => ip2 => sigmoid2
            [sigmoid2, label] => loss => sink
        })

        let solver = try! SGDSolver(net: net, device: device, batchSize: batchSize, stepCount: 1, initialLearningRate: 0.5, learningRateSchedule: { $0.0 })
        let expecation = self.expectationWithDescription("Net forward pass 1")
        solver.stepAction = { snapshot in
            let sinkData = [Float](snapshot.inputOfLayer(sink)!)
            sink.consume(Blob(sinkData))

            let sigmoid2Result = snapshot.outputOfLayer(sigmoid2)!
            let sigmoid2Expected: [Float] = [0.75136507, 0.772928465]
            for i in 0..<outputSize {
                XCTAssertEqualWithAccuracy(sigmoid2Result[i], sigmoid2Expected[i], accuracy: 0.0001)
            }

            let lossResult = sink.data[0]
            let lossExpected: Float = 0.298371109
            XCTAssertEqualWithAccuracy(lossResult, lossExpected, accuracy: 0.0001)

            let ip1ExpectedWeights: [Float] = [0.149780716, 0.24975114, 0.19956143, 0.29950229]
            let ip2ExpectedWeights: [Float] = [0.35891648, 0.511301270, 0.408666186, 0.561370121]

            let ip1ActualWeights: [Float] = arrayFromBuffer(ip1.weightsBuffer!.metalBuffer!)
            let ip2ActualWeights: [Float] = arrayFromBuffer(ip2.weightsBuffer!.metalBuffer!)

            for i in 0..<4 {
                XCTAssertEqualWithAccuracy(ip1ExpectedWeights[i], ip1ActualWeights[i], accuracy: 0.0001)
                XCTAssertEqualWithAccuracy(ip2ExpectedWeights[i], ip2ActualWeights[i], accuracy: 0.0001)
            }
            expecation.fulfill()
        }
        solver.train({ })

        self.waitForExpectationsWithTimeout(2) { error in
            if let error = error {
                XCTFail("Net.forward() failed: \(error)")
            }
        }
    }

    func testTrainXOR() {
        let inputSize = 2
        let ip1OutputSize = 10
        let outputSize = 2
        let labelSize = 2
        let batchSize = 4

        let ip1Weights = Matrix<Float>(rows: inputSize, columns: ip1OutputSize, elements: [0.841357, -0.0743405, -0.148653, 0.809335, -0.405325, 0.232792, 0.890925, 0.296268, -0.551077, 0.693504, 0.898171, -0.103595, -0.8814, 0.570545, 0.492356, 0.202726, 0.733716, -0.041736, -0.293404, 0.944786])
        let ip1Biases = ValueArray<Float>(count: ip1OutputSize, initializer: { 0.0 })
        let ip2Weights = Matrix<Float>(rows: ip1OutputSize, columns: outputSize, elements: [0.372618, -0.13119, 0.581761, -0.0753717, -0.433551, -0.524526, -0.937622, 0.17232, -0.125824, -0.892625, -0.981873, 0.659142, -0.0842931, 0.0855929, 0.0557265, 0.418915, 0.717256, -0.813372, -0.245902, 0.402122])
        let ip2Biases = ValueArray<Float>(count: outputSize, initializer: { 0.0 })

        let inputs = Matrix<Float>(rows: batchSize, columns: inputSize)
        let labels = Matrix<Float>(rows: batchSize, columns: outputSize)
        inputs[0, 0] = 0
        inputs[0, 1] = 0
        inputs[1, 0] = 0
        inputs[1, 1] = 1
        inputs[2, 0] = 1
        inputs[2, 1] = 1
        inputs[3, 0] = 1
        inputs[3, 1] = 0

        labels[0, 0] = 0
        labels[1, 0] = 1
        labels[2, 0] = 0
        labels[3, 0] = 1

        labels[0, 1] = 1
        labels[1, 1] = 0
        labels[2, 1] = 1
        labels[3, 1] = 0

        let data = Source(name: "data", data: inputs.elements, batchSize: batchSize)
        let label = Source(name: "label", data: labels.elements, batchSize: batchSize)
        let ip1 = InnerProductLayer(weights: ip1Weights, biases: ip1Biases, name: "ip1")
        let relu = ReLULayer(size: ip1OutputSize, name: "relu")
        let ip2 = InnerProductLayer(weights: ip2Weights, biases: ip2Biases, name: "ip2")

        let loss = L2LossLayer(size: labelSize, name: "loss")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: batchSize)

        let net = Net.build({
            data => ip1 => relu => ip2
            [ip2, label] => loss => sink
        })

        let expectation = self.expectationWithDescription("Net train")
        let learningRateSchedule: (Double, Int) -> Double = { lr, step in
            return lr / Double(10 * (1 + step / 500))
        }
        let solver = try! SGDSolver(net: net, device: device, batchSize: batchSize, stepCount: 1000, initialLearningRate: 0.1, learningRateSchedule: learningRateSchedule)

        var output = [Float]()
        var lossVal: Float = FLT_MAX
        solver.stepAction = { snapshot in
            if solver.currentStep % 10 == 0 {
                let newLossVal = sum([Float](snapshot.outputOfLayer(loss)!))
                XCTAssertGreaterThanOrEqual(lossVal,  newLossVal)
                lossVal = newLossVal
            }
            output = [Float](snapshot.outputOfLayer(ip2)!)
        }
        solver.train({ expectation.fulfill() })


        self.waitForExpectationsWithTimeout(20) { error in
            XCTAssertLessThan(output[0], output[4])
            XCTAssertGreaterThan(output[1], output[5])
            XCTAssertLessThan(output[2], output[6])
            XCTAssertGreaterThan(output[3], output[7])
        }
    }
}
