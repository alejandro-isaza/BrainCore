// Copyright © 2015 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import XCTest
import Metal

class MetalTestCase: XCTestCase {
    var metalLibrary: MTLLibrary {
        continueAfterFailure = false
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Failed to create a Metal device")
            fatalError("Failed to create a Metal device")
        }

        let bundle = NSBundle(forClass: self.dynamicType)
        let path = bundle.pathForResource("default", ofType: "metallib")!
        let library = try! device.newLibraryWithFile(path)
        continueAfterFailure = true

        return library
    }
}
