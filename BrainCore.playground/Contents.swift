import BrainCore
import Surge

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
ip.weights = Matrix<Double>(rows: 2, columns: 1, elements: [2, 4])
ip.biases = [1]
let sink = Sink()

let sourceRef = net.addLayer(source)
let ipRef = net.addLayer(ip)
let reluRef = net.addLayer(ReLULayer(size: 1))
let sinkRef = net.addLayer(sink)

net.connectLayer(sourceRef, toLayer: ipRef)
net.connectLayer(ipRef, toLayer: reluRef)
net.connectLayer(reluRef, toLayer: sinkRef)
net.forward()

sink.data
