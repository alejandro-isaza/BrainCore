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

### Network Definition

Before you build your network, start by building all the layers. This is as simple as calling each constructor:

```swift
let dataLayer = MyDataLayer()
let lstmLayer = LSTMLayer(weights: lstmWeights, biases: lstmBiases)
let ipLayer = InnerProductLayer(weights: ipWeights, biases: ipBiases)
let reluLayer = ReLULayer(size: ipBiases.count)
let sinkLayer = MySinkLayer()
```

**BrainCore** uses overloaded operators to make network definitions more concise. To connect layers together simply use the `=>` operator inside a `Net.build {}` closure: 

```swift
let net = Net.build {
    dataLayer => lstmLayer => ipLayer => reluLayer => sinkLayer
}
```

If you need to concatenate the output of two layers put them inside square brackets:

```swift
let net = Net.build {
    [dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer => sinkLayer
}
```

Similarly, if you need to split the output of one layer put its target layers in square brackets:

```swift
let net = Net.build {
    dataLayer => lstmLayer => ipLayer => reluLayer => [sinkLayer1, sinkLayer2]
}
```

When splitting, the `inputSize` of the target layers will determine where to split. If the sum of the target layers' `inputSize`s doesn't match the source layer's `outputSize` and error will be thrown.


If you want to continue on separate branches after a split you have to split the definition into separate lines:
```swift
let net = Net.build {
    dataLayer => lstmLayer => [ipLayer1, ipLayer2]
    ipLayer1 => reluLayer1 => sinkLayer1
    ipLayer2 => reluLayer2 => sinkLayer2
}
```

Finally if you want send multiple copies of the output of a layer to different layers use the `=>>` operator:
```swift
let net = Net.build {
    dataLayer => lstmLayer
    lstmLayer =>> ipLayer1 => reluLayer1 => sinkLayer1
    lstmLayer =>> ipLayer2 => reluLayer2 => sinkLayer2
}
```

### Evaluating

Currently **BrainCore** only supports executing pre-trained networks. Ideally you would train your network on a server using one of the well-established neural network frameworks and import the trained weights into BrainCore. We are working on implementing solvers so that you can do everything inside **BrainCore**, stay posted.

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

Next we'll build the net.

```swift
let net = Net.build {
    dataLayer => lstmLayer => ipLayer => reluLayer => sinkLayer
}
```

And finally execute! You need to provide a Metal device to the runner which is usually just the default device. 

```swift
let device = MTLCreateSystemDefaultDevice()!
let evaluator = try! Evaluator(net: net, device: device)
evaluator.evaluate { snapshot in
    // You will get notified here when the network is done the forward pass
}
```

The evaluator may fail to build if there is any problem creating the buffers or initializing all the Metal code, that's why there is a `try`.

Calling `evaluate()` will execute a single forward pass, but you can call this as often as you want. In fact you will want to call `evaluate()` multiple times before you get any results back so that you maximise the GPU bandwidth.

Your data layer will most likely want to provide new data every time you call `evaluate()`. So your code may look something like

```swift
while !shouldStop {
    dataLayer.gather()
    evaluator.evaluate(completion)
}
```

Also both the sink layer's `consume()` function and the completion closure will be called from a background thread. Make sure you synchronize access to the data as needed and try not to block on either of those calls for too long.

## Contributing

BrainCore is still under early development. Any contribution is appreciated!

---

## License

Upsurge is available under the MIT license. See the LICENSE file for more info.
