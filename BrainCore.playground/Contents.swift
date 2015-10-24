import BrainCore
import Upsurge

//: ## Helper classes and functions

//: Define a DataLayer that returns a static piece of data
class Source : DataLayer {
    var data: Blob
    init(data: Blob) {
        self.data = data
    }
}

//: Define a SinkLayer that stores the last piece of data it got
class Sink : SinkLayer {
    var data: Blob = []
    func consume(input: Blob) {
        data = input
    }
}

//: ## Network definition
let net = Net()

let source = Source(data: [1, 1])
let ip = InnerProductLayer(inputSize: 2, outputSize: 1)
ip.weights = RealMatrix(rows: 2, columns: 1, elements: [2, 4])
ip.biases = RealMatrix(rows: 1, columns: 1, elements: [1])
let sink = Sink()

let sourceRef = net.addLayer(source, name: "source")
let ipRef = net.addLayer(ip, name: "ip")
let reluRef = net.addLayer(ReLULayer(size: 1), name: "relu")
let sinkRef = net.addLayer(sink, name: "sink")

net.connectLayer(sourceRef, toLayer: ipRef)
net.connectLayer(ipRef, toLayer: reluRef)
net.connectLayer(reluRef, toLayer: sinkRef)

//: ## Network forward pass
net.forward()
sink.data
