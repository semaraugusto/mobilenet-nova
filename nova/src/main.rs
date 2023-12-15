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
use serde_with::with_prefix;

type G1 = pasta_curves::pallas::Point;
// type G2 = pasta_curves::vesta::Point;
const CIRCUIT_INPUT_F: &str = "../backbone1_test.json";

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
        private_input.insert(
            "dw_conv_remainder".to_string(),
            json!(self.dw_conv.remainder),
        );

        println!("-> dw_conv inputs inserted");

        private_input.insert("dw_bn_a".to_string(), json!(self.dw_bn.a));
        private_input.insert("dw_bn_b".to_string(), json!(self.dw_bn.b));
        private_input.insert("dw_bn_out".to_string(), json!(self.dw_bn.out));
        private_input.insert("dw_bn_remainder".to_string(), json!(self.dw_bn.remainder));
        println!("-> dw_bn inputs inserted");
        //
        private_input.insert("pw_conv_weights".to_string(), json!(self.pw_conv.weights));
        private_input.insert("pw_conv_bias".to_string(), json!(self.pw_conv.bias));
        private_input.insert("pw_conv_out".to_string(), json!(self.pw_conv.out));
        private_input.insert(
            "pw_conv_remainder".to_string(),
            json!(self.pw_conv.remainder),
        );
        println!("-> pw_conv inputs inserted");

        private_input.insert("pw_bn_a".to_string(), json!(self.pw_bn.a));
        private_input.insert("pw_bn_b".to_string(), json!(self.pw_bn.b));
        private_input.insert("pw_bn_out".to_string(), json!(self.pw_bn.out));
        private_input.insert("pw_bn_remainder".to_string(), json!(self.pw_bn.remainder));

        println!("-> pw_bn inputs inserted");
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
// #[derive(Debug, Deserialize)]
// struct BackboneLayer {
//     // dims: [Height x Width x nChannels]
//     inp: Vec<Vec<Vec<String>>>,
//     // dims: [Height x Width x nFilters ] (nChannels ommited due to depthwise convolution)
//     weights: Vec<Vec<Vec<String>>>,
//     // dims: [nFilters]
//     bias: Vec<String>,
//     // dims: [Height x Width x nFilters]
//     out: Vec<Vec<Vec<String>>>,
//     // dims: [Height x Width x nFilters]
//     remainder: Vec<Vec<Vec<String>>>,
// }

fn read_json(f: &str) -> CircuitLayer {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}

// fn generate_circuit_inputs(iteration_count) -> Vec< {
fn generate_circuit_inputs(iteration_count: usize) -> CircuitInputs {
    let mut private_inputs = Vec::new();
    let input = read_json(CIRCUIT_INPUT_F);

    println!("- Json read");

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

    let mut testing = Vec::new();
    let mut test = HashMap::new();
    // private_input.insert("in".to_string(), json!(input.image));
    test.insert("in".to_string(), json!(input.inp));
    test.insert(
        "dw_conv_weights".to_string(),
        json!(input.backbone[0].dw_conv.weights),
    );
    test.insert(
        "dw_conv_bias".to_string(),
        json!(input.backbone[0].dw_conv.bias),
    );
    test.insert(
        "dw_conv_out".to_string(),
        json!(input.backbone[0].dw_conv.out),
    );
    test.insert(
        "dw_conv_remainder".to_string(),
        json!(input.backbone[0].dw_conv.remainder),
    );

    println!("- dw_conv inputs inserted");

    test.insert("dw_bn_a".to_string(), json!(input.backbone[0].dw_bn.a));
    test.insert("dw_bn_b".to_string(), json!(input.backbone[0].dw_bn.b));
    test.insert("dw_bn_out".to_string(), json!(input.backbone[0].dw_bn.out));
    test.insert(
        "dw_bn_remainder".to_string(),
        json!(input.backbone[0].dw_bn.remainder),
    );
    println!("- dw_bn inputs inserted");
    //
    test.insert(
        "pw_conv_weights".to_string(),
        json!(input.backbone[0].pw_conv.weights),
    );
    test.insert(
        "pw_conv_bias".to_string(),
        json!(input.backbone[0].pw_conv.bias),
    );
    test.insert(
        "pw_conv_out".to_string(),
        json!(input.backbone[0].pw_conv.out),
    );
    test.insert(
        "pw_conv_remainder".to_string(),
        json!(input.backbone[0].pw_conv.remainder),
    );
    println!("- pw_conv inputs inserted");

    test.insert("pw_bn_a".to_string(), json!(input.backbone[0].pw_bn.a));
    test.insert("pw_bn_b".to_string(), json!(input.backbone[0].pw_bn.b));
    test.insert("pw_bn_out".to_string(), json!(input.backbone[0].pw_bn.out));
    test.insert(
        "pw_bn_remainder".to_string(),
        json!(input.backbone[0].pw_bn.remainder),
    );
    println!("- pw_bn inputs inserted");

    testing.push(test);

    assert!(private_inputs.len() == iteration_count);
    assert!(testing.len() == 1);
    assert!(
        testing[0] == private_inputs[0],
        "testing not equal priv inputs"
    );

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
    let iteration_count = 3;
    let root = current_dir().unwrap();

    let circuit_file = root.join(circuit_filepath);
    let r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(circuit_file));
    let witness_generator_file = root.join(witness_gen_filepath);

    let circuit_input = generate_circuit_inputs(iteration_count);

    println!("- circuit inputs generated");
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
    // let group_name = "pasta";

    // let circuit_filepath = format!("examples/toy/{}/toy.r1cs", group_name);
    // for witness_gen_filepath in [
    //     format!("examples/toy/{}/toy_cpp/toy", group_name),
    //     // format!("examples/toy/{}/toy_js/toy.wasm", group_name),
    // ] {
    let circuit_filepath = "../circuits/backbone/backbone.r1cs".to_string();
    for witness_gen_filepath in [
        "../circuits/backbone/backbone_cpp/backbone".to_string(),
        // format!("examples/toy/{}/toy_js/toy.wasm", group_name),
    ] {
        run_test(circuit_filepath.clone(), witness_gen_filepath);
    }
}

mod tests {
    use super::*;
    #[test]
    fn test() {
        println!(
            "Running test with witness generator: {} and group: {}",
            "../circuits/backbone/backbone_cpp/backbone",
            std::any::type_name::<G1>()
        );
        let input = read_json(CIRCUIT_INPUT_F);
        let mut layer = HashMap::new();
        // private_input.insert("in".to_string(), json!(input.image));
        layer.insert("in".to_string(), json!(input.inp));
        layer.insert(
            "dw_conv_weights".to_string(),
            json!(input.backbone[0].dw_conv.weights),
        );
        layer.insert(
            "dw_conv_bias".to_string(),
            json!(input.backbone[0].dw_conv.bias),
        );
        layer.insert(
            "dw_conv_out".to_string(),
            json!(input.backbone[0].dw_conv.out),
        );
        layer.insert(
            "dw_conv_remainder".to_string(),
            json!(input.backbone[0].dw_conv.remainder),
        );
        layer.insert("dw_bn_a".to_string(), json!(input.backbone[0].dw_bn.a));
        layer.insert("dw_bn_b".to_string(), json!(input.backbone[0].dw_bn.b));
        layer.insert("dw_bn_out".to_string(), json!(input.backbone[0].dw_bn.out));
        layer.insert(
            "dw_bn_remainder".to_string(),
            json!(input.backbone[0].dw_bn.remainder),
        );

        layer.insert(
            "pw_conv_weights".to_string(),
            json!(input.backbone[0].pw_conv.weights),
        );
        layer.insert(
            "pw_conv_bias".to_string(),
            json!(input.backbone[0].pw_conv.bias),
        );
        layer.insert(
            "pw_conv_out".to_string(),
            json!(input.backbone[0].pw_conv.out),
        );
        layer.insert(
            "pw_conv_remainder".to_string(),
            json!(input.backbone[0].pw_conv.remainder),
        );

        layer.insert("pw_bn_a".to_string(), json!(input.backbone[0].pw_bn.a));
        layer.insert("pw_bn_b".to_string(), json!(input.backbone[0].pw_bn.b));
        layer.insert("pw_bn_out".to_string(), json!(input.backbone[0].pw_bn.out));
        layer.insert(
            "pw_bn_remainder".to_string(),
            json!(input.backbone[0].pw_bn.remainder),
        );

        println!("- Json read");

        let mut private_input = HashMap::new();
        for i in 0..1 {
            let circuit_in = match i {
                0 => &input.inp,
                _ => &input.backbone[i - 1].pw_bn.out,
            };

            private_input.insert("in".to_string(), json!(circuit_in));

            input.backbone[0].insert_onto(&mut private_input);
            // private_inputs.push(private_input);
        }
        println!("private input len: {:?}", private_input.len());
        println!("private input keys: {:?}", private_input.keys());
        println!("layer   input len: {:?}", layer.len());
        println!("layer   input keys: {:?}", layer.keys());
        assert_eq!(layer.len(), private_input.len());
        assert_eq!(layer, private_input);
    }
}
