//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Upsurge

public typealias Blob = [Double]

public protocol Layer {
}

public protocol DataLayer : Layer {
    var data: Blob { get }
}

public protocol ForwardLayer : Layer {
    var outputSize: Int { get }
    func forward(input: Blob, inout output: Blob)
}

public protocol SinkLayer : Layer {
    func consume(input: Blob)
}
