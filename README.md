# B hat estimator in Rust

(c) 2025, Carlos Aya-Moreno

Example implementation of the B hat estimator in Rust / Burn.

Using [Burn](https://burn.dev/) as it seems a promising alternative to Python's torch,
although this estimator is _not_ a traditional NN model.

Implements the estimator in [paper.pdf](paper.pdf), equations (3) and (4).

1. Install Rust and Cargo

2. Run `cargo build` to compile

3. Run `cargo run` to run it

Currently, produces the following output approximating Cauchy distributed data with a `Cauchy(loc, scale)` distribution. The original parameters are `loc=20`, `scale=3`.

```
Starting params
Loc: 40.092420902366754
Scale: 1.6019572135408218

BHat: -0.4739512193447051 (0)
BHat: -0.5357026866098471 (0)
BHat: -0.585517988366678 (0)
BHat: -0.6281636390476865 (0)
BHat: -0.6660476247887002 (0)
BHat: -0.7004166312070627 (0)
BHat: -0.7319335792622382 (0)
BHat: -0.7609551824354809 (0)
BHat: -0.7876777455133943 (0)
BHat: -0.8122246157350027 (1)
BHat: -0.8347021263983276 (1)
BHat: -0.8552320559436145 (1)
BHat: -0.8739638810644907 (1)
BHat: -0.891070978622773 (1)
BHat: -0.9067368386630751 (1)
BHat: -0.9211377212590852 (1)
BHat: -0.9344267548285099 (1)
BHat: -0.9467217271834216 (1)
BHat: -0.9580956969797487 (1)
BHat: -0.968566045010823 (2)
BHat: -0.9780743575786447 (2)
BHat: -0.9864509664765992 (2)
BHat: -0.9933781964281085 (2)
BHat: -0.9984274855011857 (2)
BHat: -1.0013153765060199 (2)
BHat: -1.0023506807689038 (2)
BHat: -1.002461306514035 (2)
BHat: -1.002429735678315 (2)
BHat: -1.0024466075617122 (2)
BHat: -1.0024642401989383 (3)
BHat: -1.0024677442868615 (3)
BHat: -1.0024673449021815 (3)
BHat: -1.0024674319524491 (3)
BHat: -1.002467679331227 (3)
BHat: -1.0024677475309838 (3)
BHat: -1.002467743611254 (3)
BHat: -1.0024677439868686 (3)
BHat: -1.0024677476235269 (3)

End params (iterations=389)
Loc: 19.94976771826074
Scale: 2.938206710394832
```

Location seems Ok but the scale parameter needs more data points (above, 10000) to have
a reasonable approximation. This seems related to the fact that the terms are asymptotically
independent but not independent. Should I modify the estimator to achieve this?
