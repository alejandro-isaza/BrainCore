//  Copyright © 2015 Venture Media Labs. All rights reserved.

import Foundation
import Upsurge

public class Net {
    public typealias LayerRef = Int

    class Node : Hashable {
        let id: Int
        let layer: Layer
        var inputNodes = [Node]()
        var outputNodes = [Node]()
        var input: RealArray?
        var output: RealArray?

        init(layer: Layer, id: Int) {
            self.layer = layer
            self.id = id
        }

        var hashValue: Int {
            return id
        }
    }

    var nextID = 0
    var dataNodes = [Node]()
    var nodes = [LayerRef: Node]()

    var openNodes = [Node]()
    var closedNodes = Set<Node>()

    public init() {
    }

    public func addLayer(layer: Layer) -> LayerRef {
        validateLayer(layer)
        let node = Node(layer: layer, id: nextID++)

        if layer is DataLayer {
            dataNodes.append(node)
        }
        nodes[node.id] = node

        return node.id
    }

    private func validateLayer(layer: Layer) {
        let dataLayer = layer is DataLayer
        let forwardLayer = layer is ForwardLayer
        let sinkLayer = layer is SinkLayer
        precondition(dataLayer || forwardLayer || sinkLayer, "Layer has to be one of: data, forward, sink")
        precondition(!dataLayer || !forwardLayer, "Layer can't be both a data layer and a forward layer")
        precondition(!sinkLayer || !forwardLayer, "Layer can't be both a sink layer and a forward layer")
        precondition(!dataLayer || !sinkLayer, "Layer can't be both a data layer and a sink layer")
    }

    public func connectLayer(layer1: LayerRef, toLayer layer2: LayerRef) {
        guard let node1 = nodes[layer1] else {
            precondition(false, "Layer not found")
            return
        }
        guard let node2 = nodes[layer2] else {
            precondition(false, "Layer not found")
            return
        }

        node1.outputNodes.append(node2)
        node2.inputNodes.append(node1)
    }

    public func forward() {
        openNodes.removeAll(keepCapacity: true)
        closedNodes.removeAll(keepCapacity: true)

        // Collect all data
        for n in dataNodes {
            let dataLayer = n.layer as! DataLayer
            n.output = dataLayer.data
            closeNode(n)
        }

        while !openNodes.isEmpty {
            let node = openNodes.popLast()!
            if closedNodes.contains(node) {
                continue
            }

            let data = collectDataForNode(node)
            if let forwardLayer = node.layer as? ForwardLayer {
                setupOutputData(node, size: forwardLayer.outputSize)
                forwardLayer.forward(data, output: &node.output!)
                assert(node.output!.count == forwardLayer.outputSize)
            } else if let sinkLayer = node.layer as? SinkLayer {
                sinkLayer.consume(data)
            }

            closeNode(node)
        }
    }

    private func closeNode(node: Node) {
        closedNodes.insert(node)
        let newOpenNodes = node.outputNodes.filter{ isNodeReady($0) }
        openNodes.appendContentsOf(newOpenNodes)
    }

    private func isNodeReady(node: Node) -> Bool {
        for n in node.inputNodes {
            if !closedNodes.contains(n) {
                return false
            }
        }
        return true
    }

    private func collectDataForNode(node: Node) -> RealArray {
        var size = 0
        for n in node.inputNodes {
            size += n.output?.count ?? 0
        }

        if node.input == nil || node.input!.capacity < size {
            node.input = RealArray(count: size, repeatedValue: 0.0)
        }

        var i = 0
        for n in node.inputNodes {
            if let output = n.output {
                node.input!.replaceRange(i..<i+output.count, with: output)
                i += output.count
            }
        }

        return node.input!
    }

    private func setupOutputData(node: Node, size: Int) -> RealArray {
        if node.output == nil || node.output!.capacity > size {
            node.output = RealArray(capacity: size)
        }
        
        node.output!.append(to: size - 1, with: 0.0)
        
        return node.output!
    }
}

func == (lhs: Net.Node, rhs: Net.Node) -> Bool {
    return lhs.id == rhs.id
}
