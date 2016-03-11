# BrainCore

[![CocoaPods Compatible](https://img.shields.io/cocoapods/v/BrainCore.svg)](https://img.shields.io/cocoapods/v/BrainCore.svg)
[![Carthage Compatible](https://img.shields.io/badge/Carthage-compatible-4BC51D.svg?style=flat)](https://github.com/Carthage/Carthage)

BrainCore is a simple but fast neural network framework written in Swift. It uses [Metal](https://developer.apple.com/metal/) which makes it screamin' fast.


## Features

- [x] Inner product layers
- [x] Linear rectifier (ReLU) layers 
- [x] LSTM layers


## Requirements

- iOS 8.0+ / Mac OS X 10.11+
- Xcode 7.2+
- A device that supports Metal (doesn't work on the iOS simulator)

## Usage

Currently **BrainCore** only supports executing pre-trained networks. Ideally you would train your network on a server using one of the well-established neural network frameworks and import the trained weights into BrainCore. We are working on implementing solvers so that you can do everything inside **BrainCore**, stay posted.

Start by building a network.

```swift
let net = Net()
```

The network build process is divided into three parts: create the layers, create the buffers between layers and connect everything together. The reason you have to explicitly provide the buffers is so that you have more flexibility on how data moves through the network. For instance you can concatenate by specifying different offsets of the same buffer or you can reuse the output of a layer for multiple subnets.

Let's start by creating the layers.

```swift
// Load weights and biases from a pre-trained network
let lstmWeights = ...
let lstmBiases = ...
let ipWeights = ...
let ipBiases = ...

// Create layers
let dataLayer = MyDataLayer()
let lstmLayer = LSTMLayer(weights: lstmWeights, biases: lstmBiases)
let ipLayer = InnerProductLayer(weights: ipWeights, biases: ipBiases)
let reluLayer = ReLULayer(size: ipBiases.count)
let sinkLayer = MySinkLayer()
```

Everything here is straightforward. Keep in mind that you have to provide your own data and sink layers. The responsibility of the data layer is provide the inputs to the network. You might get this from a file or from a real-time feed. And the sink layer is where you process the output of the network.

Now let's create the buffers. In this case, and in most cases, there is a  single buffer between layers.

```swift
// Set up buffers
let input = net.addBufferWithName("input", size: lstmLayer.inputSize)
let lstmOutput = net.addBufferWithName("lstmOutput", size: lstmLayer.outputSize)
let ipOutput = net.addBufferWithName("ipOutput", size: ipLayer.outputSize)
let output = net.addBufferWithName("output", size: reluLayer.outputSize)
```

Next we connect everything together. This is the part that is a bit painful right now, but we are working on making this easier.

```swift
// Make connections
let dataLayerRef = net.addLayer(dataLayer, name: "data")
net.connectLayer(dataLayerRef, toBuffer: input)

let lstmLayerRef = net.addLayer(lstmLayer, name: "lstm")
net.connectBuffer(input, atOffset: 0, toLayer: lstmLayerRef)
net.connectLayer(lstmLayerRef, toBuffer: lstmOutput)

let ipLayerRef = net.addLayer(ipLayer, name: "ip")
net.connectBuffer(lstmOutput, atOffset: 0, toLayer: ipLayerRef)
net.connectLayer(ipLayerRef, toBuffer: ipOutput)

let reluLayerRef = net.addLayer(reluLayer, name: "ReLU")
net.connectBuffer(ipOutput, atOffset: 0, toLayer: reluLayerRef)
net.connectLayer(reluLayerRef, toBuffer: output)

let sinkLayerRef = net.addLayer(sinkLayer, name: "sink")
net.connectBuffer(output, atOffset: 0, toLayer: sinkLayerRef)
```

And finally execute! You need to provide a Metal device to the runner which is usually just the default device. 

```swift
let device = MTLCreateSystemDefaultDevice()!
let runner = try! Runner(net: net, device: device)
runner.forwardPassAction = {
	// You will get notified here when the network is done a forward pass
}
runner.forward()
```

The runner may fail to build if there is any problem creating the buffers or initializing all the Metal code, that's why there is a `try`.

Calling `forward()` will execute a single forward pass, but you can call this as often as you want. In fact you will want to call `forward()` multiple times before you get any results back so that you maximise the GPU bandwidth.

Your data layer will most likely want to provide new data every time you call `forward()`. So your code may look something like

```swift
while !shouldStop {
	dataLayer.gather()
	runner.forward()
}
```

Also both the sink layer's `consume()` function and the `forwardPassAction` will be called from a background thread. Make sure you synchronize access to the data as needed and try not to block on either of those calls for too long.

## Contributing

BrainCore is still under early development. Any contribution is appreciated!

---

## License

Upsurge is available under the MIT license. See the LICENSE file for more info.
