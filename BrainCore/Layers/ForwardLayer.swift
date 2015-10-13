//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Surge

public protocol ForwardLayer {
    func forward(input: Matrix<Double>, inout output: Matrix<Double>)
}
