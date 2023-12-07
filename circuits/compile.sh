PTAU_PATH=./powersOfTau28_hez_final_24.ptau

compile () {
    local circuit_name="$1"
    echo "$circuit_name.circom"
    mkdir -p $circuit_name &&
    circom --r1cs --wasm --sym -c -o $circuit_name "$circuit_name".circom --prime vesta &&
    echo -e "Done!\n"
    cd $circuit_name/"$circuit_name"_cpp &&
    make && 
    cd -
}

compile_phase2 () {
    local circuit="$1" outdir="$1" pathToCircuitDir="$1"
    echo $outdir;
    mkdir -p $outdir;

    echo "Setting up Phase 2 ceremony for $circuit"
    echo "Outputting circuit_final.zkey and verifier.sol to $outdir"

    npx snarkjs groth16 setup "$pathToCircuitDir/$circuit.r1cs" $PTAU_PATH "$outdir/circuit_final.zkey" &&
    # echo "test" | npx snarkjs zkey contribute "$outdir/circuit_0000.zkey" "$outdir/circuit_0001.zkey" --name"1st Contributor name" -v &&
    # npx snarkjs zkey verify "$pathToCircuitDir/$circuit.r1cs" $PTAU_PATH "$outdir/circuit_0001.zkey" &&
    # npx snarkjs zkey beacon "$outdir/circuit_0000.zkey" "$outdir/circuit_final.zkey" 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 10 -n="Final Beacon phase2" &&
    npx snarkjs zkey verify "$pathToCircuitDir/$circuit.r1cs" $PTAU_PATH "$outdir/circuit_final.zkey" &&
    npx snarkjs zkey export verificationkey "$outdir/circuit_final.zkey" "$outdir/verification_key.json" &&

    # npx snarkjs zkey export solidityverifier "$outdir/circuit_final.zkey" $outdir/verifier.sol
    echo "Done!\n"
}

# compile layer1
# compile layer1 &&
# compile_phase2 layer1 layer1 layer1

compile $1
# compile $1 &&
# compile_phase2 $1
# compile Conv2D &&
# compile_phase2 Conv2D Conv2D Conv2D 
