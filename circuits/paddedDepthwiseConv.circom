pragma circom 2.1.1;

include "./node_modules/circomlib/circuits/sign.circom";
include "./node_modules/circomlib/circuits/bitify.circom";
include "./node_modules/circomlib/circuits/mux1.circom";
include "./node_modules/circomlib/circuits/comparators.circom";
include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./util.circom";

// Depthwise Convolution layer with 'same' padding
// Note that nFilters must be a multiple of nChannels
// n = 10 to the power of the number of decimal places
// component main = DepthwiseConv2D(34, 34, 8, 8, 3, 1);
template PaddedDepthwiseConv2D (paddedNRows, paddedNCols, nChannels, nFilters, kernelSize, strides, n) {
    var outRows = (paddedNRows-kernelSize)\strides+1;
    var outCols = (paddedNCols-kernelSize)\strides+1;

    var padding = 1;
    signal input in[paddedNRows][paddedNCols][nChannels];
    signal input weights[kernelSize][kernelSize][nFilters]; // weights are 3d because depth is 1
    signal input bias[nFilters];
    signal input remainder[paddedNRows][paddedNCols][nFilters];

    signal input out[paddedNRows][paddedNCols][nFilters];
    // signal input out[outRows][outCols][nFilters];

    component mul[outRows][outCols][nFilters];
    component elemSum[outRows][outCols][nFilters];
    component muxes[outRows][outCols][nFilters];
    component is_equal[outRows][outCols][nFilters];

    var valid_groups = nFilters % nChannels;
    var filtersPerChannel = nFilters / nChannels;

    // Can probably remove below loops using main loop instead
    // Checking if padding has been applied correctly on the columns (albeit walking along the rows and checking if them are 0)
    for (var row=0; row<paddedNRows; row++) {
        for (var filterMultiplier=1; filterMultiplier<=filtersPerChannel; filterMultiplier++) {
            for (var channel=0; channel<nChannels; channel++) {
                var filter = filterMultiplier*channel;
                out[row][0][filter] === 0;
                out[row][paddedNCols-1][filter] === 0;

                remainder[row][0][filter] === 0;
                remainder[row][paddedNCols-1][filter] === 0;
            }
        }
    }
    // Checking if padding has been applied correctly on the rows (albeit walking along the columns and checking if them are 0)
    for (var col=0; col<paddedNCols; col++) {
        for (var filterMultiplier=1; filterMultiplier<=filtersPerChannel; filterMultiplier++) {
            for (var channel=0; channel<nChannels; channel++) {
                var filter = filterMultiplier*channel;
                out[0][col][filter] === 0;
                out[paddedNRows-1][col][filter] === 0;

                remainder[row][0][filter] === 0;
                remainder[row][paddedNCols-1][filter] === 0;
            }
        }
    }

    for (var row=0; row<outRows; row++) {
        for (var col=0; col<outCols; col++) {
            for (var filterMultiplier=1; filterMultiplier<=filtersPerChannel; filterMultiplier++) {
                for (var channel=0; channel<nChannels; channel++) {
                    var filter = filterMultiplier*channel;

                    mul[row][col][filter] = matElemMul(kernelSize,kernelSize);

                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            mul[row][col][filter].a[x][y] <== in[row*strides+x][col*strides+y][channel];
                            mul[row][col][filter].b[x][y] <== weights[x][y][filter];
                        }
                    }

                    elemSum[row][col][filter] = matElemSum(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            elemSum[row][col][filter].a[x][y] <== mul[row][col][filter].out[x][y];
                        }
                    }
                    assert (remainder[row+1][col+1][filter] <= n);

                    is_equal[row][col][filter] = IsEqual();
                    is_equal[row][col][filter].in[0] <== remainder[row+1][col+1][filter];
                    is_equal[row][col][filter].in[1] <== n;

                    muxes[row][col][filter] = Mux1();
                    muxes[row][col][filter].c[0] <== elemSum[row][col][filter].out + bias[filter];
                    muxes[row][col][filter].c[1] <== 0;
                    muxes[row][col][filter].s <== is_equal[row][col][filter];


                    // out[row+1][col+1][filter] * n + remainder[row+1][col+1][filter] === elemSum[row][col][filter].out + bias[filter];
                    out[row+1][col+1][filter] * n + remainder[row+1][col+1][filter] === muxes[row][col][filter].out
                }
            }
        }
    }
}
component main = PaddedDepthwiseConv2D(34, 34, 256, 256, 3, 1);
