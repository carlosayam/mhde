use mhde;

use burn::backend::{Autodiff, NdArray};
use argparse::{ArgumentParser, Store, StoreOption, StoreTrue};

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use rand::distributions::Distribution;
use statrs::distribution::Cauchy;


struct Options {
    loc: f64,
    scale: f64,
    num: usize,
    seed: Option<u64>,
    split: bool,
}

fn set_options(options: &mut Options) {
    let mut parser = ArgumentParser::new();

    parser.set_description("Minimum Helliger Distance Estimator for Cauchy-distributed 1D sample and model");

    parser.refer(&mut options.loc)
        .add_argument("loc", Store, "Location of Cauchy distribution to generate a sample (def 0.0)");

    parser.refer(&mut options.scale)
    .add_argument("scale", Store, "Scale of Cauchy distribution to generate a sample (def 1.0)");

    parser.refer(&mut options.num)
        .add_argument("num", Store, "Number of observations (def 1000)");

    parser.refer(&mut options.seed)
        .add_option(&["-s", "--seed"], StoreOption, "Provide a seed for reproducibility, otherwise random sample");

    parser.refer(&mut options.split)
        .add_option(&["--split"], StoreTrue, "Use split sample variant");

    parser.parse_args_or_exit();
}

/// Generates Cauchy distributed sample 
fn generate(options: &Options) -> Vec<f64> {
    let mut rng: ChaCha8Rng = match options.seed {
        Some(val) => ChaCha8Rng::seed_from_u64(val),
        None => ChaCha8Rng::from_entropy(),
    };

    // create random vec
    let dist: Cauchy = Cauchy::new(options.loc, options.scale).expect("Wrong parameters for Cauchy distribution");
    let vec = Vec::from_iter((0..options.num).map(|_| dist.sample(&mut rng)));
    vec
}

fn main() {
    let mut options = Options { loc: 0.0, scale: 1.0, num: 1000, seed: None, split: false };
    set_options(&mut options);

    let vec = generate(&options);

    mhde::run::<Autodiff<NdArray<f64, i64>>>(
        vec,
        options.split,
        Default::default(),
    );
}
