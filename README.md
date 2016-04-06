# BrainCore

[![CocoaPods Compatible](https://img.shields.io/cocoapods/v/BrainCore.svg)](https://img.shields.io/cocoapods/v/BrainCore.svg)
[![Carthage Compatible](https://img.shields.io/badge/Carthage-compatible-4BC51D.svg?style=flat)](https://github.com/Carthage/Carthage)

BrainCore is a simple but fast neural network framework written in Swift. It uses [Metal](https://developer.apple.com/metal/) which makes it screamin' fast.


## Features

- [x] Inner product layers
- [x] Linear rectifier (ReLU) layers
- [x] Sigmoid layers
- [x] LSTM layers
- [x] L2 Loss layers


## Requirements

- iOS 8.0+ / Mac OS X 10.11+
- Xcode 7.2+
- A device that supports Metal (doesn't work on the iOS simulator)

## Usage

Currently **BrainCore** only supports executing pre-trained networks. Ideally you would train your network on a server using one of the well-established neural network frameworks and import the trained weights into BrainCore. We are working on implementing solvers so that you can do everything inside **BrainCore**, stay posted.

Let's start by creating the layers.

```swift
// Load weights and biases from a pre-trained network
let lstmWeights = ...
let lstmBiases = ...
let ipWeights = ...
let ipBiases = ...

// Create layers
let dataLayer1 = MyDataLayer()
let dataLayer2 = MyDataLayer()
let lstmLayer = LSTMLayer(weights: lstmWeights, biases: lstmBiases)
let ipLayer = InnerProductLayer(weights: ipWeights, biases: ipBiases)
let reluLayer = ReLULayer(size: ipBiases.count)
let sinkLayer = MySinkLayer()
```

Next we'll build the net. Square brackets (`[]`) indicate a concatenation of the contained layer's outputs.

```swift
let net = [dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer => sinkLayer
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
