# tnorms

[![crates.io](https://img.shields.io/crates/v/tnorms.svg)](https://crates.io/crates/tnorms)
[![Documentation](https://docs.rs/tnorms/badge.svg)](https://docs.rs/tnorms)

T-norm and t-conorm families.

`tnorms` provides common fuzzy-logic aggregation families and fold helpers for
multi-way conjunction and disjunction over values in `[0, 1]`.

Dual-licensed under MIT or Apache-2.0.

```toml
[dependencies]
tnorms = "0.2.0"
```

```rust
use tnorms::{tconorm, tnorm, PRODUCT};

let or_degree = tconorm(PRODUCT, 0.5, 0.6);
let and_degree = tnorm(PRODUCT, 0.5, 0.6);

assert!((or_degree - 0.8).abs() < 1e-12);
assert!((and_degree - 0.3).abs() < 1e-12);
```

## Families

Named constants:

| Constant | T-norm | T-conorm |
|----------|--------|----------|
| `GODEL` | `min(a, b)` | `max(a, b)` |
| `PRODUCT` | `a * b` | `a + b - a * b` |
| `LUKASIEWICZ` | `max(0, a + b - 1)` | `min(1, a + b)` |

`LogicFamily` also provides the corresponding residual implication:

```rust
let implication = tnorms::LogicFamily::Product.residuum(0.7, 0.28);
assert!((implication - 0.4).abs() < 1e-12);
```

`TruthAlgebra` exposes the same logics as marker types for callers that select
the algebra at the type level:

```rust
use tnorms::{Product, TruthAlgebra};

let degree = Product::tnorm_f32(0.4, 0.7);
assert!((degree - 0.28).abs() < 1e-6);
```

Core logic families also expose scalar gradients:

```rust
let gradient = tnorms::LogicFamily::Product.tnorm_gradient(0.4, 0.7);
assert_eq!((gradient.da, gradient.db), (0.7, 0.4));
```

For long product aggregations, keep the value in log space until the final
readout:

```rust
let log_degree = tnorms::log_product(&[0.9, 0.8, 0.7], 1e-12);
let degree = tnorms::product_from_log(log_degree);
assert!((degree - 0.504).abs() < 1e-12);
```

Families also expose conservative copula metadata:

```rust
assert_eq!(tnorms::PRODUCT.is_copula(), Some(true));
assert_eq!(tnorms::TConormFamily::Drastic.is_copula(), Some(false));
```

With the `burn` feature enabled, the same core families can evaluate Burn
tensors:

```rust
use burn::tensor::{backend::Backend, Tensor};

fn product_and<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    tnorms::LogicFamily::Product.tnorm_tensor(a, b)
}
```

The `burn` feature follows Burn 0.20's Rust toolchain requirements.

| Family | T-conorm behavior |
|--------|-------------------|
| `Maximum` | Hard OR |
| `Probabilistic` | Probabilistic sum |
| `Bounded` | Lukasiewicz bounded sum |
| `Einstein` | Smooth rational OR |
| `Hamacher` | Aggressive rational OR |
| `Yager` | Lp-style family |
| `Frank` | Interpolating family |
| `Dombi` | Power-based family |
| `SchweizerSklar` | Parametric family spanning min, product, Lukasiewicz, and drastic limits |
| `SugenoWeber` | Parametric family spanning drastic, Lukasiewicz, and product limits |
| `AczelAlsina` | Product-to-min Archimedean family |
| `MayorTorrens` | Min-to-Lukasiewicz ordinal-sum family |
| `Drastic` | Discontinuous drastic sum |
| `NilpotentMinimum` | Nilpotent-minimum dual |

`GeneratorFamily` exposes additive generators for the Archimedean families.
`residuum()` computes the corresponding residual implication where the family
has a left-continuous path. `OrdinalSum` composes families on disjoint
sub-intervals of `[0, 1]`.

Additional aggregation helpers include standard/Sugeno/Yager negations,
Kleene-Dienes/Reichenbach/Lukasiewicz/Godel/Product implications, OWA, power
means, and representable uninorms.

Default builds use `std`. `--no-default-features` builds the scalar API without
`std`; the Burn feature requires `std`.

## Examples

```sh
cargo run --example logic_table
cargo run --example aggregation_recipe
```
