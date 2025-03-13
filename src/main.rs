mod bhat;

use burn::backend::{Autodiff, NdArray};

fn main() {
    bhat::run::<Autodiff<NdArray<f64, i64>>>(Default::default());
}
