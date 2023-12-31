// based of https://github.com/nalinbhardwaj/Nova-Scotia/blob/main/examples/toy_pasta.rs
use std::{
    collections::HashMap, env::current_dir, fs::File, io::BufReader, path::PathBuf, str::FromStr,
    time::Instant,
};

use nova_scotia::{
    circom::{
        circuit::{CircomCircuit, R1CS},
        reader::{generate_witness_from_wasm, load_r1cs},
    },
    create_public_params, create_recursive_circuit, FileLocation, C1, C2, F, S,
};
use nova_snark::{traits::Group, CompressedSNARK, PublicParams};
use num_bigint::BigInt;
use num_traits::Num;
use primitive_types::U256;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use serde_with::with_prefix;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

const CIRCUIT_INPUT_F: &str = "../test_inputs/nova_backbone_input.json";
const MIMC_CIRCUIT_WASM: &str = "../circuits/MiMC3D/MiMC3D_js/MiMC3D.wasm";
const MIMC_CIRCUIT_R1CS: &str = "../circuits/MiMC3D/MiMC3D.r1cs";

#[derive(Debug)]
struct CircuitInputs {
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: [F<G1>; 2],
    // start_public_dummy: Option<Vec<G2>>,
}

#[derive(Debug, Deserialize)]
struct DepthConvLayer {
    // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
    weights: Vec<Vec<Vec<String>>>,
    // dims: [nFilters]
    bias: Vec<String>,
    // dims: [Height x Width x nFilters]
    out: Vec<Vec<Vec<String>>>,
    // dims: [Height x Width x nFilters]
    remainder: Vec<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct PointConvLayer {
    // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
    weights: Vec<Vec<String>>,
    // dims: [nFilters]
    bias: Vec<String>,
    // dims: [Height x Width x nFilters]
    out: Vec<Vec<Vec<String>>>,
    // dims: [Height x Width x nFilters]
    remainder: Vec<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct BatchNormLayer {
    // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
    // dims: [nFilters]
    a: Vec<String>,
    // dims: [nFilters]
    b: Vec<String>,
    // dims: [Height x Width x nFilters]
    out: Vec<Vec<Vec<String>>>,
    // dims: [Height x Width x nFilters]
    remainder: Vec<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct BackboneLayer {
    // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
    #[serde(flatten, with = "prefix_dw_conv")]
    dw_conv: DepthConvLayer,
    #[serde(flatten, with = "prefix_dw_bn")]
    dw_bn: BatchNormLayer,
    #[serde(flatten, with = "prefix_pw_conv")]
    pw_conv: PointConvLayer,
    #[serde(flatten, with = "prefix_pw_bn")]
    pw_bn: BatchNormLayer,
}

impl BackboneLayer {
    fn insert_onto(&self, private_input: &mut HashMap<String, Value>) {
        private_input.insert("dw_conv_weights".to_string(), json!(self.dw_conv.weights));
        private_input.insert("dw_conv_bias".to_string(), json!(self.dw_conv.bias));
        private_input.insert("dw_conv_out".to_string(), json!(self.dw_conv.out));
        private_input.insert("dw_bn_a".to_string(), json!(self.dw_bn.a));
        private_input.insert("dw_bn_b".to_string(), json!(self.dw_bn.b));
        private_input.insert("dw_bn_out".to_string(), json!(self.dw_bn.out));

        private_input.insert("pw_conv_weights".to_string(), json!(self.pw_conv.weights));
        private_input.insert("pw_conv_bias".to_string(), json!(self.pw_conv.bias));
        private_input.insert("pw_conv_out".to_string(), json!(self.pw_conv.out));

        private_input.insert("pw_bn_a".to_string(), json!(self.pw_bn.a));
        private_input.insert("pw_bn_b".to_string(), json!(self.pw_bn.b));
        private_input.insert("pw_bn_out".to_string(), json!(self.pw_bn.out));
    }
}

with_prefix!(prefix_dw_conv "dw_conv_");
with_prefix!(prefix_dw_bn "dw_bn_");
with_prefix!(prefix_pw_conv "pw_conv_");
with_prefix!(prefix_pw_bn "pw_bn_");

#[derive(Debug, Deserialize)]
struct CircuitLayer {
    // dims: [Height x Width x nChannels]
    inp: Vec<Vec<Vec<String>>>,
    backbone: Vec<BackboneLayer>,
}

fn read_json(f: &str) -> CircuitLayer {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}
/*
 * Taken from https://github.com/lyronctk/zator/blob/main/nova/src/main.rs
 * Computes the MiMC hash of an input 3D array. Used to satisfy input hash
 * check for initial backbone layer.
 */
#[derive(Serialize)]
struct MiMC3DInput {
    dummy: String,
    arr: Vec<Vec<Vec<String>>>,
}
pub type F1 = <G1 as Group>::Scalar;

fn mimc3d(r1cs: &R1CS<F1>, wasm: PathBuf, arr: Vec<Vec<Vec<String>>>) -> BigInt {
    println!("Start hashing");
    let witness_gen_output = PathBuf::from("circom_witness.wtns");

    let inp = MiMC3DInput {
        dummy: String::from("0"),
        arr: arr.clone(),
    };
    let input_json = serde_json::to_string(&inp).unwrap();
    let witness = generate_witness_from_wasm::<<G1 as Group>::Scalar>(
        &FileLocation::PathBuf(wasm),
        &input_json,
        &witness_gen_output,
    );
    println!("MIMC Witness generated");

    let circuit = CircomCircuit {
        r1cs: r1cs.clone(),
        witness: Some(witness),
    };
    let pub_outputs = circuit.get_public_outputs();
    std::fs::remove_file(witness_gen_output).unwrap();

    let stripped = format!("{:?}", pub_outputs[0])
        .strip_prefix("0x")
        .unwrap()
        .to_string();

    let big = BigInt::from_str_radix(&stripped, 16).unwrap();
    println!("mimc3d output: {:?}", big);

    big
}

// fn generate_circuit_inputs(iteration_count) -> Vec< {
fn generate_circuit_inputs(
    iteration_count: usize,
    mimc_r1cs: &R1CS<F1>,
    mimc_wasm: PathBuf,
) -> CircuitInputs {
    let mut private_inputs = Vec::new();
    let input = read_json(CIRCUIT_INPUT_F);

    println!("- Json read");

    let step_in = mimc3d(mimc_r1cs, mimc_wasm, input.inp.clone());

    for i in 0..iteration_count {
        let mut private_input = HashMap::new();
        let circuit_in = match i {
            0 => &input.inp,
            _ => &input.backbone[i - 1].pw_bn.out,
        };

        private_input.insert("in".to_string(), json!(circuit_in));

        input.backbone[i].insert_onto(&mut private_input);

        private_inputs.push(private_input);
    }

    assert!(private_inputs.len() == iteration_count);

    let v_1 = step_in.to_str_radix(10);
    // let start_public_input = vec![
    let start_public_input = [
        F1::from(0),
        F1::from_raw(U256::from_dec_str(&v_1).unwrap().0),
    ];

    CircuitInputs {
        private_inputs,
        start_public_input,
    }
}

fn run_test(circuit_filepath: String, witness_gen_filepath: String) {
    // type G1 = pasta_curves::pallas::Point;
    // type G2 = pasta_curves::vesta::Point;

    println!(
        "Running test with witness generator: {} and group: {}",
        witness_gen_filepath,
        std::any::type_name::<G1>()
    );
    let iteration_count = 13;
    let root = current_dir().unwrap();

    let mimc3d_r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(root.join(MIMC_CIRCUIT_R1CS)));
    let mimc3d_wasm = root.join(MIMC_CIRCUIT_WASM);
    println!("MIMC3D R1CS and WASM loaded",);

    let circuit_input = generate_circuit_inputs(iteration_count, &mimc3d_r1cs, mimc3d_wasm);
    println!("- circuit inputs generated");

    let circuit_file = root.join(circuit_filepath);
    let r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(circuit_file));
    let witness_generator_file = root.join(witness_gen_filepath);
    println!("backbone R1CS and WASM loaded",);

    let start = Instant::now();
    let pp: PublicParams<G1, G2, _, _> = create_public_params(r1cs.clone());

    println!("PublicParams creation took {:?}", start.elapsed());

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
    let circuit_filepath = "../circuits/backbone/backbone.r1cs".to_string();
    for witness_gen_filepath in ["../circuits/backbone/backbone_cpp/backbone".to_string()] {
        run_test(circuit_filepath.clone(), witness_gen_filepath);
    }
}
