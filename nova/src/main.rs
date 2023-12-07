// based of https://github.com/nalinbhardwaj/Nova-Scotia/blob/main/examples/toy_pasta.rs
use std::{collections::HashMap, env::current_dir, fs::File, io::BufReader, time::Instant};

use nova_scotia::{
    circom::reader::load_r1cs, create_public_params, create_recursive_circuit, FileLocation, F, S,
};
use nova_snark::{
    // provider,
    // traits::{circuit::StepCircuit, Group},
    CompressedSNARK,
    PublicParams,
};
use serde::Deserialize;
use serde_json::{json, Value};

type G1 = pasta_curves::pallas::Point;
// type G2 = pasta_curves::vesta::Point;
const CIRCUIT_INPUT_F: &str = "../backbone_input.json";

#[derive(Debug)]
struct CircuitInputs {
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: [F<G1>; 2],
    // start_public_dummy: Option<Vec<G2>>,
}

#[derive(Debug, Deserialize)]
struct BackboneLayer {
    // dims: [Height x Width x nChannels]
    inp: Vec<Vec<Vec<String>>>,
    // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
    weights: Vec<Vec<Vec<String>>>,
    // dims: [nFilters]
    bias: Vec<String>,
    // dims: [Height x Width x nFilters]
    out: Vec<Vec<Vec<String>>>,
    // dims: [Height x Width x nFilters]
    remainder: Vec<Vec<Vec<String>>>,
}

fn read_json(f: &str) -> BackboneLayer {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}

// fn generate_circuit_inputs(iteration_count) -> Vec< {
fn generate_circuit_inputs() -> CircuitInputs {
    let mut private_inputs = Vec::new();
    // for i in 0..iteration_count {
    // let mut private_input = HashMap::new();
    // private_input.insert("adder".to_string(), json!(i));
    // private_inputs.push(private_input);
    // }
    let input = read_json(CIRCUIT_INPUT_F);

    let mut private_input = HashMap::new();
    // private_input.insert("in".to_string(), json!(input.image));
    private_input.insert("in".to_string(), json!(input.inp));
    private_input.insert("dw_conv_weights".to_string(), json!(input.weights));
    private_input.insert("dw_conv_bias".to_string(), json!(input.bias));
    private_input.insert("dw_conv_out".to_string(), json!(input.out));
    private_input.insert("dw_conv_remainder".to_string(), json!(input.remainder));

    private_inputs.push(private_input);

    // println!("input: {:?}", input);
    // println!("Private inputs: {:?}", private_inputs);

    let start_public_input = [F::<G1>::from(10), F::<G1>::from(10)];
    CircuitInputs {
        private_inputs,
        start_public_input,
    }
}

fn run_test(circuit_filepath: String, witness_gen_filepath: String) {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    println!(
        "Running test with witness generator: {} and group: {}",
        witness_gen_filepath,
        std::any::type_name::<G1>()
    );
    let iteration_count = 1;
    let root = current_dir().unwrap();

    let circuit_file = root.join(circuit_filepath);
    let r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(circuit_file));
    let witness_generator_file = root.join(witness_gen_filepath);

    let circuit_input = generate_circuit_inputs();

    let pp: PublicParams<G1, G2, _, _> = create_public_params(r1cs.clone());

    println!(
        "Number of constraints per step (primary circuit): {}",
        pp.num_constraints().0
    );
    println!(
        "Number of constraints per step (secondary circuit): {}",
        pp.num_constraints().1
    );

    println!(
        "Number of variables per step (primary circuit): {}",
        pp.num_variables().0
    );
    println!(
        "Number of variables per step (secondary circuit): {}",
        pp.num_variables().1
    );

    println!("Creating a RecursiveSNARK...");
    let start = Instant::now();
    let recursive_snark = create_recursive_circuit(
        FileLocation::PathBuf(witness_generator_file),
        r1cs,
        circuit_input.private_inputs,
        circuit_input.start_public_input.to_vec(),
        &pp,
    )
    .unwrap();
    println!("RecursiveSNARK creation took {:?}", start.elapsed());
    //
    // TODO: empty?
    let z0_secondary = [F::<G2>::from(0)];

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let start = Instant::now();
    let res = recursive_snark.verify(
        &pp,
        iteration_count,
        &circuit_input.start_public_input,
        &z0_secondary,
    );
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res,
        start.elapsed()
    );
    assert!(res.is_ok());

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
    let start = Instant::now();

    let (pk, vk) = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::setup(&pp).unwrap();
    let res = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::prove(&pp, &pk, &recursive_snark);
    println!(
        "CompressedSNARK::prove: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(
        &vk,
        iteration_count,
        circuit_input.start_public_input.to_vec(),
        z0_secondary.to_vec(),
    );
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok());
}

fn main() {
    let group_name = "pasta";

    let circuit_filepath = format!("examples/toy/{}/toy.r1cs", group_name);
    for witness_gen_filepath in [
        format!("examples/toy/{}/toy_cpp/toy", group_name),
        // format!("examples/toy/{}/toy_js/toy.wasm", group_name),
    ] {
        run_test(circuit_filepath.clone(), witness_gen_filepath);
    }
}
