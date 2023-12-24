PTAU_PATH=./powersOfTau28_hez_final_24.ptau

compile () {
    local circuit_name="$1";
    local prime="$2";

    echo -e "compiling $circuit_name\.circom on $prime"
    mkdir -p $circuit_name &&
    circom --r1cs --wasm --sym -c -o $circuit_name "$circuit_name".circom --prime $prime &&
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
    npx snarkjs zkey verify "$pathToCircuitDir/$circuit.r1cs" $PTAU_PATH "$outdir/circuit_final.zkey" &&
    npx snarkjs zkey export verificationkey "$outdir/circuit_final.zkey" "$outdir/verification_key.json" &&

    echo "Done!\n"
}

# compile $1
# compile $1 &&
# compile_phase2 $1
