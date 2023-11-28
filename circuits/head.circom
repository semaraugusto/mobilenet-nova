pragma circom 2.1.1;

include "./pointwiseConv2D.circom";
include "./depthwiseConv2D.circom";
include "./Conv2D.circom";
include "./node_modules/circomlib-ml/circuits/BatchNormalization2D.circom";

template MobileNetCIFAR10(n) {
                // H x W x C
    var inputSize = 32;
    var paddedInputSize = 34;
    var nChannels = 3;
    var nConvFilters = 8;
    var kernelSize = 3;

    signal input in[paddedInputSize][paddedInputSize][nChannels];

    // Initial Layer: Conv2D + BN + RELU
    signal input conv2d_weights[kernelSize][kernelSize][nChannels][nConvFilters]; // H x W x C x K
    signal input conv2d_bias[nConvFilters];
    signal input conv2d_out[inputSize][inputSize][nConvFilters];
    signal input conv2d_remainder[inputSize][inputSize][nConvFilters];

    signal input bn_a[nConvFilters];
    signal input bn_b[nConvFilters];
    signal input bn_out[inputSize][inputSize][nConvFilters];
    signal input bn_remainder[inputSize][inputSize][nConvFilters];

    signal output out;
    log("something");

    // Start Initial Layer: Conv2D + BN + RELU
    // START INITIAL CONV 2D
    var stride = 1;
    component conv2d = Conv2D(paddedInputSize, paddedInputSize, nChannels, nConvFilters, kernelSize, stride, n);
    log("before conv2d");
    conv2d.in <== in;
    conv2d.weights <== conv2d_weights;
    conv2d.bias <== conv2d_bias;
    conv2d.out <== conv2d_out;
    conv2d.remainder <== conv2d_remainder;
    log("after conv2d");
    // START INITIAL BATCH NORM 2D 
    component bn = BatchNormalization2D(inputSize, inputSize, nConvFilters, n);
    bn.in <== conv2d_out;
    bn.a <== bn_a;
    bn.b <== bn_b;
    bn.out <== bn_out;
    bn.remainder <== bn_remainder;
    log("after bn");
    // END INITIAL BATCH NORM 2D 

    out <== 1; 
    log("end");
}

component main = MobileNetCIFAR10(10**15);
