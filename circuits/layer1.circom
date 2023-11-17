pragma circom 2.1.1;
// include "./Conv2D.circom";

include "./node_modules/circomlib/circuits/sign.circom";
include "./node_modules/circomlib/circuits/bitify.circom";
include "./node_modules/circomlib-matrix/circuits/matElemMul.circom";
include "./node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./util.circom";

// Conv2D layer with valid padding
template Conv2D (nRows, nCols, nChannels, nFilters, kernelSize, padding, strides) {
// template Conv2D (nRows, nCols, nChannels) {
    signal input in[nRows][nCols][nChannels];
    signal input weights[kernelSize][kernelSize][nChannels][nFilters];
    var outRows = (nRows-kernelSize+2*padding)\strides+1;
    var outCols = (nCols-kernelSize+2*padding)\strides+1;
    log(9999999999999999);
    log(outRows);
    log(outCols);
    log(9999999999999999);
    signal input bias[nFilters];
    signal output out[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];
    log(in[0][0][0]);
    log(weights[0][0][0][0]);
    log(bias[0]);

    component mul[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component elemSum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component sum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];

    for (var i=0; i<(nRows-kernelSize)\strides+1; i++) {
        for (var j=0; j<(nCols-kernelSize)\strides+1; j++) {
            for (var k=0; k<nChannels; k++) {
                for (var m=0; m<nFilters; m++) {
                    mul[i][j][k][m] = matElemMul(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            mul[i][j][k][m].a[x][y] <== in[i*strides+x][j*strides+y][k];
                            mul[i][j][k][m].b[x][y] <== weights[x][y][k][m];
                        }
                    }
                    elemSum[i][j][k][m] = matElemSum(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            elemSum[i][j][k][m].a[x][y] <== mul[i][j][k][m].out[x][y];
                        }
                    }
                }
            }
            for (var m=0; m<nFilters; m++) {
                sum[i][j][m] = Sum(nChannels);
                for (var k=0; k<nChannels; k++) {
                    sum[i][j][m].in[k] <== elemSum[i][j][k][m].out;
                }
                out[i][j][m] <== sum[i][j][m].out + bias[m];
            }
        }
    }
    log(out[0][0][0]);
    log(-1);
    // component bits = Num2Bits(256);
    // bits.in <== out[0][0][0];
    // component sign = Sign();
    //
    // for (var i = 2; i < 256; i++) {
    //     sign.in[i] <== bits.out[i];
    // }
    // log(11111111111111111111111);
    // log(sign.sign);
}

// component main { public [step_in] } = Conv2D(32, 32, 3, 64, 3, 1);
component main = Conv2D(32, 32, 3, 32, 3, 1, 1);
// component main = Conv2D(32, 32, 3);
