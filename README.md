# ZK Mobile net

This is a mobile net for ZK, it is a light weight neural network for ZK. It is based on the [MobileNet](https://arxiv.org/abs/1704.04861) architecture. 

Initial implementation in [PyTorch](https://pytorch.org/). 

Circuit implementation (TBD) in [Circom](https://docs.circom.io/) and [Nova](https://github.com/microsoft/Nova) (TBD).
# Organization
* Pytorch implementation in "./mobilenet.ipynb" and "models/mobilenet.py"
* Initial Circom implementation in "./circuits/" (only separable convolution for now. Will add more circuits soon)

# Scaling inputs for ZK
Circom only acceps field elements as values while pytorch uses floats. We therefore need a quantization method for running the model in ZK.

The image input and model weights have been quantized using a multiplication by 10**EXPONENT being exponent the precision of the operations.

# Future work
Pytorch already has a quantization method implemented as described in their [docs](https://pytorch.org/docs/stable/quantization.html) and in [this paper](https://arxiv.org/pdf/1712.05877.pdf). 
Making the circuits compatible with the pytorch quantization method would be a good next step.
