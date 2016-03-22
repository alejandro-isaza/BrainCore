// Copyright © 2016 Venture Media Labs. All rights reserved.
//
// This file is part of BrainCore. The full BrainCore copyright notice,
// including terms governing use, modification, and redistribution, is
// contained in the file LICENSE at the root of the source code distribution
// tree.

import Metal

public protocol SolverParameters {}

public protocol Solver {
    var updateFunctionName: String { get }
    
    var net: Net { get }
    var batchSize: Int { get }
    var endStep: Int { get }
    
    var params: SolverParameters { get }
    var loss: Double { get }
    
    var runner: Runner { get }
    
    func train()
}

