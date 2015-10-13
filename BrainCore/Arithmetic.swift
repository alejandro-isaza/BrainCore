//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Accelerate
import Foundation
import Surge

/// Matrix multiplication with recycled result matrix
public func mul(lhs: Matrix<Double>, _ rhs: Matrix<Double>, inout result: Matrix<Double>) {
    assert(lhs.columns == rhs.rows, "Matrix inner dimensions should match")
    assert(result.rows == lhs.rows, "Invalid result blob")
    assert(result.columns == rhs.columns, "Invalid result blob")

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(lhs.rows), Int32(rhs.columns), Int32(lhs.columns),
        1.0,
        lhs.elements, Int32(lhs.columns),
        rhs.elements, Int32(rhs.columns),
        0.0,
        &(result.elements), Int32(result.columns))
}
