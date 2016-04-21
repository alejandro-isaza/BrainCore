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

### Net Definition Syntax
- Square brackets (`[]`) indicate a concatenation when on the the left, and splitting when on the right.
- `=>` indicates a connection between layers;
  - From the first unattached output element of the left side, to the right side input.
- `=>>` indicates a full output connection between layers; 
  - Regardless of whether the left side's output has been attached, this will connect to the beginning of the output.
```swift 
let net = Net.build {
	[dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer
	reluLayer => sinkLayer1 // Here sinkLayer1 consumes reluLayer.output[0..<sinkLayer1.inputSize]
	reluLayer => sinkLayer2 // and sinkLayer2 consumes reluLayer.output[sinkLayer1.inputSize..<sinkLayer1.inputSize + sinkLayer2.inputSize]
}
```
is equivalent to:
```swift 
let net = Net.build {
	[dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer => [sinkLayer1, sinkLayer2]
}
```
is **NOT** equivalent to:
```swift 
let net = Net.build {
	[dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer
	reluLayer =>> sinkLayer1
	reluLayer =>> sinkLayer2 // Here both sinkLayers consume the same data
}
```


### Training
Let's start by creating the layers.

```swift
// Load weights and biases from a pre-trained network or just initialize to zeros
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
let sinkLayer1 = MySinkLayer()
let sinkLayer2 = MySinkLayer()
```

Next we'll build the net. 

```swift
let net = Net.build {
	[dataLayer1, dataLayer2] => lstmLayer => ipLayer => reluLayer
	reluLayer =>> sinkLayer1
	reluLayer =>> sinkLayer2
}
```

And finally execute! You need to provide a Metal device to the runner which is usually just the default device. 

```swift
let device = MTLCreateSystemDefaultDevice()!
let trainer = try! Trainer(net: net, device: device, batchSize: 15)
trainer.run = { snapshot in
	// You will get notified here at the end of every step
}
```

### Evaluating
Sometimes you may have a pre-trained model that you just want to fetch output from after providing it data. This is the use-case of the `Evaluator`

```swift
let net = Net.build {
	data => ip => sink
}
let evaluator = try! Evaluator(net: net, device: device)
while !shouldStop {
	data.gather()
	evaluator.evaluate { snapshot in
		let output = sink.data
		// Here you can use the output data as you please
	}
}
```

## Contributing

BrainCore is still under early development. Any contribution is appreciated!

---

## License

Upsurge is available under the MIT license. See the LICENSE file for more info.
