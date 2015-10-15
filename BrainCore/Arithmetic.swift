//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Accelerate
import Foundation
import Surge

/// Row-vector and matrix multiplication
public func mul(lhs: [Double], _ rhs: Matrix<Double>, inout result: [Double]) {
    assert(lhs.count == rhs.rows, "Matrix inner dimensions should match")
    assert(result.count == rhs.columns, "Invalid result vector")

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(1), Int32(rhs.columns), Int32(lhs.count),
        1.0,
        lhs, Int32(lhs.count),
        rhs.elements, Int32(rhs.columns),
        0.0,
        &result, Int32(result.count))
}


/// Vector addition with reusable result
public func add(lhs: [Double], _ rhs: [Double], inout result: [Double]) {
    assert(lhs.count == rhs.count, "Vector sizes should match")
    assert(result.count == rhs.count, "Invalid result vector")
    vDSP_vaddD(lhs, 1, rhs, 1, &result, 1, vDSP_Length(lhs.count))
}
