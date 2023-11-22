// pragma circom 2.0.0;
pragma circom 2.1.1;

// include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
// include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./pointwiseConv2D.circom";
include "./depthwiseConv2D.circom";
// include "./util.circom";

// Separable convolution layer with valid padding
template SeparableConv2D (nRows, nCols, nChannels, nDepthFilters, kernelSize, strides) {
    var outRows = (nRows-kernelSize)\strides+1;
    var outCols = (nCols-kernelSize)\strides+1;

    signal input in[nRows][nCols][nChannels];
    signal input depthWeights[kernelSize][kernelSize][nDepthFilters];
    signal input depthBias[nDepthFilters];
    signal output out[outRows][outCols][nDepthFilters];

    component depthConv = DepthwiseConv2D(nRows, nCols, nChannels, nDepthFilters, kernelSize, strides);
    // component pointConv = PontwiseConv2D(nRows, nCols, nChannels, nPointFilters, strides);

    for (var x=0; x<kernelSize; x++) {
        for (var y=0; y<kernelSize; y++) {
            for (var filter=0; filter<nDepthFilters; filter++) {
                depthConv.weights[x][y][filter] <== depthWeights[x][y][filter];
            }
        }
    }
    for (var filter=0; filter<nDepthFilters; filter++) {
        depthConv.bias[filter] <== depthBias[filter];
    }
    for (var row=0; row < nRows; row++) {
        for (var col=0; col < nCols; col++) {
            for (var channel=0; channel < nChannels; channel++) {
                depthConv.in[row][col][channel] <== in[row][col][channel];
            }
        }
    }
    for (var row=0; row < outRows; row++) {
        for (var col=0; col < outCols; col++) {
            for (var filter=0; filter < nDepthFilters; filter++) {
                out[row][col][filter] <== depthConv.out[row][col][filter];

            }
        }
    }
}
component main = SeparableConv2D(34, 34, 8, 8, 3, 1);
