#!/bin/bash

circom ./toy.circom -l ../../../circuits --r1cs --wasm --sym --c --output ./pasta/ --prime vesta
cd pasta/toy_cpp && make
cd -

# circom ./toy.circom --r1cs --wasm --sym --c --output ./bn254/ --prime bn128
# cd bn254/toy_cpp && make
