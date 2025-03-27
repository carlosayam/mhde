# B hat estimator in Rust

(c) 2025, Carlos Aya-Moreno

Example implementation of the B hat estimator in Rust / Burn.

Using [Burn](https://burn.dev/) as it seems a promising alternative to Python's torch,
although this estimator is _not_ a traditional NN model.

Implements the estimator in paper.pdf, equations (3) and (4).

At the moment, it does not work.

1. Install Rust and Cargo

2. Run `cargo build` to compile

3. Run `cargo run` to run it

Currently, produces the following output

```
Starting val
Loc: 30.047816198115136
Scale: 0.8864389687591542
BHat: -0.6149687233691192 (0)
BHat: -0.721162539450469 (0)
BHat: -0.7928300035043044 (0)
BHat: -0.8441609204633068 (0)
BHat: -0.8828385628732693 (0)
BHat: -0.9129690699338049 (0)
BHat: -0.9367846332313989 (0)
BHat: -0.9556174574192026 (0)
BHat: -0.970363013494028 (0)
BHat: -0.9816883952716301 (1)
BHat: -0.9901352028271135 (1)
BHat: -0.9961765414382825 (1)
BHat: -1.0002476183171782 (1)
BHat: -1.0027628249163656 (1)
BHat: -1.00412809284881 (1)
BHat: -1.004738306667952 (1)
BHat: -1.0049407135477455 (1)
BHat: -1.0049804378989287 (1)
BHat: -1.0049811670562192 (1)
BHat: -1.0049807221389562 (2)
BHat: -1.0049822110618216 (2)
BHat: -1.0049833792416525 (2)
BHat: -1.0049837611762595 (2)
BHat: -1.0049837979709286 (2)
BHat: -1.0049837883809924 (2)
BHat: -1.0049837925878087 (2)
BHat: -1.004983798821189 (2)
BHat: -1.0049838009965577 (2)
Starting end (iters=284)
Loc: 20.00930054231659
Scale: 2.677800595406431
```

Location seems Ok but the scale parameter does not seem right.
