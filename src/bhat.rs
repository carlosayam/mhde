use burn::{
    module::{
        Module,
        Param,
        ParamId,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::{
        Tensor,
        Backend,
        Config,
    },
    tensor::backend::AutodiffBackend,
};

use std::f64::consts::PI;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use rand::distributions::Distribution;
use statrs::distribution::Cauchy;
use linfa_nn::{BallTree, distance::L1Dist, NearestNeighbour};
use ndarray::{Array, array};

#[derive(Module, Debug)]
pub struct BHatModel<B: Backend> {
    loc: Param<Tensor<B, 1>>,
    scale: Param<Tensor<B, 1>>,
}

impl<B: AutodiffBackend> BHatModel<B> {
    pub fn forward(&self, data: &Tensor<B, 1>, balls: &Tensor<B, 1>) -> Tensor<B, 1> {
        // calculate Pdf_{Cauchy(l,s)}(data) =
        // \frac{1}{\pi s (1 + (\frac{x - l}{s})^2)}; l = loc, s = scale
        let v = (self.loc.val() - data.clone()) / self.scale.val();
        let v = v.powi_scalar(2);
        let v = (v + 1.0) * self.scale.val() * PI;
        let pdf = v.powi_scalar(-1);
        let v = (pdf * balls.clone()).powf_scalar(0.5);
        v.sum()
    }
}

fn calculate_balls<B: Backend>(data: &Vec<f64>, device: &B::Device) -> Tensor<B, 1> {
    let num = data.len();

    let algo = BallTree::new();
    let arr = Array::from_shape_vec([num, 1], data.clone()).unwrap();
    let arr = arr.view();

    let nn_index = algo.from_batch(&arr, L1Dist).unwrap();

    let radii: Vec<f64> = data.iter()
        .map(|pt: &f64| (nn_index.k_nearest((array![*pt]).view(), 2).unwrap(), pt))
        .map(|resp: (Vec<(ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>, usize)>, &f64)|
                    (resp.1 - resp.0[1].0[0]).abs())  // distance to nearest neighbour
        .map(|v: f64| v * 2.0)                        // ball volume in dimension 1
        .collect();

    Tensor::from_data(radii.as_slice(), device)
}

fn min_median_max(numbers: &Vec<f64>) -> (f64, f64, f64) {

    let mut to_sort = numbers.clone();
    to_sort.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mid = numbers.len() / 2;
    let med = if numbers.len() % 2 == 0 {
        (numbers[mid - 1] + numbers[mid]) / 2.0
    } else {
        numbers[mid]
    };
    (to_sort[0], med, to_sort[numbers.len()-1])
}

#[derive(Config)]
pub struct TrainingConfig {

    #[config(default = 100)]
    pub num_runs: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-2)]
    pub lr: f64,

    pub config_optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // some global refs
    let config_optimizer = AdamConfig::new();
    let config = TrainingConfig::new(config_optimizer);

    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(config.seed);
    let num: usize = 10;

    // create random vec
    let dist: Cauchy = Cauchy::new(5.0, 2.0).unwrap();
    let vec = Vec::from_iter((0..num).map(|_| dist.sample(&mut rng)));

    let (v_min, v_med, v_max) = min_median_max(&vec);
    let balls = calculate_balls::<B>(&vec, &device);
    let factor = -2.0 / ((num as f64) * PI).sqrt();   // makes negative so we minimize BHat

    let loc = Tensor::from_data([v_med], &device);
    let scale = Tensor::from_data([v_max - v_min], &device);
    let data = Tensor::from_data(vec.as_slice(), &device);

    let mut model = BHatModel {
        loc: Param::initialized(ParamId::new(), loc),
        scale: Param::initialized(ParamId::new(), scale),
    };
    println!("Starting val");
    println!("Loc: {}", model.loc.val().clone().into_scalar());
    println!("Scale: {}", model.scale.val().clone().into_scalar());

    let mut optimizer = config.config_optimizer.init();

    for ix in 1..config.num_runs + 1 {

        let bhat = model.forward(&data, &balls) * factor;

        let grads = bhat.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(config.lr, model, grads);
        if ix % 10 == 0 {
            println!("BHat: {}", bhat.into_scalar());
        }

    }

    println!("Starting end");
    println!("Loc: {}", model.loc.val().clone().into_scalar());
    println!("Scale: {}", model.scale.val().clone().into_scalar());
}
