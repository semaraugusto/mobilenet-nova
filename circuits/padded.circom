pragma circom 2.1.1;
// include "./Conv2D.circom";

// include "./node_modules/circomlib/circuits/sign.circom";
// include "./node_modules/circomlib/circuits/bitify.circom";
// include "./node_modules/circomlib/circuits/comparators.circom";
// include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
// include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
// include "./paddedDepthwiseConv.circom";
// include "./PaddedPointwiseConv2D.circom";
// include "./BatchNormalization2D.circom";
// include "./node_modules/circomlib-ml/circuits/ReLU.circom";
// include "./util.circom";
include "./utils/utils.circom";
include "./SeparableBNConv.circom";

// Depthwise Convolution layer with valid padding
// Note that nFilters must be a multiple of nChannels
// n = 10 to the power of the number of decimal places
template Backbone(nRows, nCols, nChannels, nDepthFilters, nPointFilters, n) {
    var kernelSize = 3;
    var strides = 1;

    // // [running of hash outputted by the previous layer, hash of the activations of the previous layer]
    // signal input step_in[2];
    // signal output step_out[2];

    signal input in[nRows][nCols][nChannels];

    signal input dw_conv_weights[kernelSize][kernelSize][nDepthFilters]; // H x W x C x K
    signal input dw_conv_bias[nDepthFilters];
    signal input dw_conv_out[nRows][nCols][nDepthFilters];
    signal input dw_conv_remainder[nRows][nCols][nDepthFilters];

    signal input dw_bn_a[nDepthFilters];
    signal input dw_bn_b[nDepthFilters];
    signal input dw_bn_out[nRows][nCols][nDepthFilters];
    signal input dw_bn_remainder[nRows][nCols][nDepthFilters];

    // signal input pw_conv_weights[kernelSize][kernelSize][nPointFilters]; // H x W x C x K
    signal input pw_conv_weights[nDepthFilters][nPointFilters]; // weights are 2d because kernel_size is 1
    signal input pw_conv_bias[nPointFilters];
    signal input pw_conv_out[nRows][nCols][nPointFilters];
    signal input pw_conv_remainder[nRows][nCols][nPointFilters];

    signal input pw_bn_a[nPointFilters];
    signal input pw_bn_b[nPointFilters];
    signal input pw_bn_out[nRows][nCols][nPointFilters];
    signal input pw_bn_remainder[nRows][nCols][nPointFilters];


    component layer = SeparableBNConvolution(nRows, nCols, nChannels, nDepthFilters, nPointFilters, 10**15);
    layer.in <== in;
    layer.dw_conv_weights <== dw_conv_weights;
    layer.dw_conv_bias <== dw_conv_bias;
    layer.dw_conv_out <== dw_conv_out;
    layer.dw_conv_remainder <== dw_conv_remainder;

    layer.dw_bn_a <== dw_bn_a;
    layer.dw_bn_b <== dw_bn_b;
    layer.dw_bn_out <== dw_bn_out;
    layer.dw_bn_remainder <== dw_bn_remainder;

    layer.pw_conv_weights <== pw_conv_weights;
    layer.pw_conv_bias <== pw_conv_bias;
    layer.pw_conv_out <== pw_conv_out;
    layer.pw_conv_remainder <== pw_conv_remainder;

    layer.pw_bn_a <== pw_bn_a;
    layer.pw_bn_b <== pw_bn_b;
    layer.pw_bn_out <== pw_bn_out;
    layer.pw_bn_remainder <== pw_bn_remainder;

    // component layer1 = SeparableBNConvolution(nRows, nCols, nChannels, nDepthFilters, nPointFilters, 10**15);
    // layer1.in <== pw_bn_out;
    // layer1.dw_conv_weights <== l1_dw_conv_weights;
    // layer1.dw_conv_bias <== l1_dw_conv_bias;
    // layer1.dw_conv_out <== l1_dw_conv_out;
    // layer1.dw_conv_remainder <== l1_dw_conv_remainder;
    //
    // layer1.dw_bn_a <== l1_dw_bn_a;
    // layer1.dw_bn_b <== l1_dw_bn_b;
    // layer1.dw_bn_out <== l1_dw_bn_out;
    // layer1.dw_bn_remainder <== l1_dw_bn_remainder;
    //
    // layer1.pw_conv_weights <== l1_pw_conv_weights;
    // layer1.pw_conv_bias <== l1_pw_conv_bias;
    // layer1.pw_conv_out <== l1_pw_conv_out;
    // layer1.pw_conv_remainder <== l1_pw_conv_remainder;
    //
    // layer1.pw_bn_a <== l1_pw_bn_a;
    // layer1.pw_bn_b <== l1_pw_bn_b;
    // layer1.pw_bn_out <== l1_pw_bn_out;
    // layer1.pw_bn_remainder <== l1_pw_bn_remainder;
    // component mimc_output = MimcHashMatrix3D(nRows, nRows, nPointFilters);
    // mimc_hash_activations.matrix <== pw_bn_out;
    // step_out[1] <== mimc_hash_activations.hash;
}

// component main { public [step_in] } = Backbone(32, 32, 64, 64, 64, 10**15);
component main  = Backbone(32, 32, 64, 64, 64, 10**15);
