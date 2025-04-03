mod bhat;

use burn::backend::{Autodiff, NdArray};
use argparse::{ArgumentParser, Store, StoreOption};

struct Options {
    num: usize,
    seed: Option<u64>,
}

fn set_options(options: &mut Options) {
    let mut parser = ArgumentParser::new();

    parser.set_description("Minimum Helliger Distance Estimator for Cauchy-distributed sample and model");

    parser.refer(&mut options.num)
        .add_argument("num", Store, "Number of observations (def 1000)");

    parser.refer(&mut options.seed)
        .add_option(&["-s", "--seed"], StoreOption, "Provide a seed for reproducibility, otherwise random sample");

    parser.parse_args_or_exit();
}

fn main() {
    let mut options = Options { num: 1000, seed: None };
    set_options(&mut options);

    bhat::run::<Autodiff<NdArray<f64, i64>>>(
        options.num,
        options.seed,
        Default::default(),
    );
}
