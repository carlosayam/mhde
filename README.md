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
Sample size: 8000
Starting params
Loc: 4.725596957516934
Scale: 2.0024465169260273

BHat: -0.5812619457321596 (10)
BHat: -0.6464460745182188 (20)
BHat: -0.6988252147591788 (30)
BHat: -0.7420346321090421 (40)
BHat: -0.7784008144131196 (50)
BHat: -0.8094497493439611 (60)
BHat: -0.8362227703052266 (70)
BHat: -0.8594936704253852 (80)
BHat: -0.8798915421194071 (90)
BHat: -0.8979515530729139 (100)
BHat: -0.9141246636637286 (110)
BHat: -0.9287705010234276 (120)
BHat: -0.942146640190865 (130)
BHat: -0.9543992257690462 (140)
BHat: -0.9655557479248154 (150)
BHat: -0.9755201086452785 (160)
BHat: -0.9840734821734218 (170)
BHat: -0.9908972812756098 (180)
BHat: -0.9956669678198781 (190)
BHat: -0.998290374684303 (200)
BHat: -0.9992124346621915 (210)
BHat: -0.9993219755276769 (220)
BHat: -0.9993006001711994 (230)
BHat: -0.9993151807450186 (240)
BHat: -0.9993305479491986 (250)
BHat: -0.999334034932025 (260)
BHat: -0.9993338036658429 (270)
BHat: -0.9993338613731023 (280)
BHat: -0.9993340809322736 (290)
BHat: -0.999334147918702 (300)
BHat: -0.9993341448502138 (310)
BHat: -0.9993341450569827 (320)
BHat: -0.9993341485809026 (330)

End params (iterations=338)
Loc: 19.943927519488348
Scale: 2.971114750500754
```

Location seems Ok but the scale parameter needs more data points (above, 8000) to have
a reasonable approximation. This seems related to the fact that the terms are asymptotically
independent but not independent. Should I modify the estimator to achieve this?
