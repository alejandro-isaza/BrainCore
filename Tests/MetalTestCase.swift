// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Metal

class MetalTestCase: XCTestCase {
    var device: MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            XCTFail("Failed to create a Metal device")
            fatalError("Failed to create a Metal device")
        }

        return d
    }
}
