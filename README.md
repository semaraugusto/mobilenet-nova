# ZK Mobile net

This repo is (tentative) zero knowledge implementation of a [MobileNet](https://arxiv.org/abs/1704.04861) architecture. It is a light weight neural network for ZK created for embedded devices.

Initial implementation in [PyTorch](https://pytorch.org/).

Circuit implementation in [Nova](https://github.com/microsoft/Nova)

# Organization
* Pytorch implementation in "./mobilenet.ipynb" and "models/mobilenet.py"
* Initial Circom implementation in "./circuits/" (only separable convolution for now. Will add more circuits soon)

# Compiling the circuits
to compile the circuits you need to install circom and snarkjs. After that, go through the following steps
```bash
cd circuits;
npm install; # Downloads circomlib-ml and dependencies
sh compile.sh head # Compiles the head
sh compile.sh backbone # Compiles the backbone
sh compile.sh tail # Compiles the tail
```

# Running Nova circuit
To run the nova circuit we just need a standard `cargo run --release` with `test_inputs/nova_backbone_input.json` inplace and with the `backbone.circom` circuit properly compiled as described above.

In order to run the nova circuit we need more than 64GBs of RAM for even a 2 layer input. I believe there might be a memory leak somewhere in nova-scotia or in nova itself.

# Notebooks
Model was initially trained on the [mobilenet notebook](mobilenet.ipynb) and mobilenet-****.ipynb variants. Each tried to use a more ZK friendly activation friendly but all of them led to models which diverged.

In order to generate your own inputs to try and run through the nova circuit, you should go to [circuit_test](circuit_tests.ipynb)

# Scaling inputs for ZK
Circom only acceps field elements as values while pytorch uses floats. We therefore need a quantization method for running the model in ZK.

The image input and model weights have been quantized using a multiplication by 10**EXPONENT being exponent the precision of the operations.

# Future work
Pytorch already has a quantization method implemented as described in their [docs](https://pytorch.org/docs/stable/quantization.html) and in [this paper](https://arxiv.org/pdf/1712.05877.pdf). 

Making the circuits compatible with the pytorch quantization method would be a good next step.
