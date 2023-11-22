// pragma circom 2.0.0;
pragma circom 2.1.1;

// include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
// include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./pointwiseConv2D.circom";
include "./depthwiseConv2D.circom";
// include "./util.circom";

// Separable convolution layer with valid padding
template SeparableConv2D (nRows, nCols, nChannels, nDepthFilters, nPointFilters, kernelSize, strides) {
// template SeparableConv2D (nRows, nCols, nChannels, nPointFilters) {
    var outRows = (nRows-kernelSize)\strides+1;
    var outCols = (nCols-kernelSize)\strides+1;
    // var outRows = nRows;
    // var outCols = nCols;

    signal input in[nRows][nCols][nChannels];
    signal input depthWeights[kernelSize][kernelSize][nDepthFilters]; // weights are 3d because depth is 1
    signal input depthBias[nDepthFilters];

    signal input pointWeights[nChannels][nPointFilters]; // weights are 2d because kernelSize is one
    signal input pointBias[nPointFilters];

    signal output out[outRows][outCols][nPointFilters];

    component depthConv = DepthwiseConv2D(nRows, nCols, nChannels, nDepthFilters, kernelSize, strides);
    component pointConv = PointwiseConv2D(outRows, outCols, nDepthFilters, nPointFilters);

    for (var filter=0; filter<nDepthFilters; filter++) {
        for (var x=0; x<kernelSize; x++) {
            for (var y=0; y<kernelSize; y++) {
                depthConv.weights[x][y][filter] <== depthWeights[x][y][filter];
            }
        }
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
                // out[row][col][filter] <== depthConv.out[row][col][filter];
                pointConv.in[row][col][filter] <== depthConv.out[row][col][filter];

            }
        }
    }
    // for (var row=0; row < nRows; row++) {
    //     for (var col=0; col < nCols; col++) {
    //         for (var channel=0; channel < nChannels; channel++) {
    //             pointConv.in[row][col][channel] <== in[row][col][channel];
    //         }
    //     }
    // }
    for (var filter=0; filter < nPointFilters; filter++) {
        for (var channel=0; channel < nChannels; channel++) {
            pointConv.weights[channel][filter] <== pointWeights[channel][filter];
        }
        pointConv.bias[filter] <== pointBias[filter];
    }
    for (var row=0; row < outRows; row++) {
        for (var col=0; col < outCols; col++) {
            for (var filter=0; filter < nPointFilters; filter++) {
                // log("out[", row, "][", col, "][", filter, "] = ", pointConv.out[row][col][filter]);
                out[row][col][filter] <== pointConv.out[row][col][filter];

            }
        }
    }
}
// component main = SeparableConv2D(34, 34, 8, 8, 3, 1); // as depthwise
// component main = SeparableConv2D(32, 32, 8, 16); // as pointwise
component main = SeparableConv2D(34, 34, 8, 8, 16, 3, 1); // as depthwise

