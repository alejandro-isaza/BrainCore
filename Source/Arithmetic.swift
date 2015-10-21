//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Accelerate
import Foundation
import Upsurge

/// Row-vector and matrix multiplication
public func mul(lhs: RealArray, _ rhs: RealMatrix, inout result: RealArray) {
    assert(lhs.count == rhs.rows, "Matrix inner dimensions should match")
    assert(result.capacity == rhs.columns, "Invalid result vector")

    vDSP_mmulD(lhs.pointer, 1, rhs.pointer, 1, result.pointer, 1, 1, vDSP_Length(rhs.columns), vDSP_Length(rhs.rows))
}


/// Vector addition with reusable result
public func add(lhs: RealArray, _ rhs: RealArray, inout result: RealArray) {
    assert(lhs.count == rhs.count, "Vector sizes should match")
    assert(result.count == rhs.count, "Invalid result vector")
    vDSP_vaddD(lhs.pointer, 1, rhs.pointer, 1, result.pointer, 1, vDSP_Length(lhs.count))
}
