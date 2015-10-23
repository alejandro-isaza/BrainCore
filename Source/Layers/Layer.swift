//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Upsurge

public typealias Blob = RealArray

public protocol Layer {
}

public protocol DataLayer : Layer {
    var data: Blob { get }
}

public protocol ForwardLayer : Layer {
    var outputSize: Int { get }

    /// Forward-propagate the input blob
    func forward(input: Blob, inout output: Blob)
}

public protocol BackwardLayer : Layer {
    /// Backward-propagate the output differences
    func backward(outputDiff: RealMatrix, input: RealMatrix, inout inputDiff: RealMatrix)
}

public protocol SinkLayer : Layer {
    func consume(input: Blob)
}
