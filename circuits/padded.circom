pragma circom 2.1.1;
// include "./Conv2D.circom";

include "./node_modules/circomlib/circuits/sign.circom";
include "./node_modules/circomlib/circuits/bitify.circom";
include "./node_modules/circomlib/circuits/comparators.circom";
include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./paddedDepthwiseConv.circom";
include "./PaddedPointwiseConv2D.circom";
include "./util.circom";

// Depthwise Convolution layer with valid padding
// Note that nFilters must be a multiple of nChannels
// n = 10 to the power of the number of decimal places
template Padded (nRows, nCols, nChannels, nDepthFilters, nPointFilters, n) {
    var paddedInputSize = nRows;
    var kernelSize = 3;
    var strides = 1;


    signal input in[paddedInputSize][paddedInputSize][nChannels];

    signal input dw_conv_weights[kernelSize][kernelSize][nDepthFilters]; // H x W x C x K
    signal input dw_conv_bias[nDepthFilters];
    signal input dw_conv_out[paddedInputSize][paddedInputSize][nDepthFilters];
    signal input dw_conv_remainder[paddedInputSize][paddedInputSize][nDepthFilters];

    // signal input pw_conv_weights[kernelSize][kernelSize][nPointFilters]; // H x W x C x K
    signal input pw_conv_weights[nDepthFilters][nPointFilters]; // weights are 2d because kernel_size is 1
    signal input pw_conv_bias[nPointFilters];
    signal input pw_conv_out[paddedInputSize][paddedInputSize][nPointFilters];
    signal input pw_conv_remainder[paddedInputSize][paddedInputSize][nPointFilters];

    log("START");
    
    // component conv = DepthwiseConv2D(paddedInputSize, paddedInputSize, nChannels, nConvFilters, kernelSize, stride, 10**15);
    component dw_conv = PaddedDepthwiseConv2D(nRows, nCols, nChannels, nDepthFilters, kernelSize, strides, 10**15);

    dw_conv.in <== in;
    dw_conv.weights <== dw_conv_weights;
    dw_conv.bias <== dw_conv_bias;
    dw_conv.out <== dw_conv_out;
    dw_conv.remainder <== dw_conv_remainder;
    log("dw_conv done");

    component pw_conv = PointwiseConv2D(nRows, nCols, nDepthFilters, nPointFilters, n);
    pw_conv.in <== dw_conv_out;
    pw_conv.weights <== pw_conv_weights;
    pw_conv.bias <== pw_conv_bias;
    pw_conv.out <== pw_conv_out;
    pw_conv.remainder <== pw_conv_remainder;
    log("END");
}

component main = Padded(7, 7, 3, 3, 6, 10**15);
