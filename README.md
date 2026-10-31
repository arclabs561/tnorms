# tnorms

[![crates.io](https://img.shields.io/crates/v/tnorms.svg)](https://crates.io/crates/tnorms)
[![Documentation](https://docs.rs/tnorms/badge.svg)](https://docs.rs/tnorms)

T-norm and t-conorm families.

`tnorms` provides common fuzzy-logic aggregation families and fold helpers for
multi-way conjunction and disjunction over values in `[0, 1]`.

Dual-licensed under MIT or Apache-2.0.

```toml
[dependencies]
tnorms = "0.2.1"
```

```rust
use tnorms::{LogicFamily, Product, TNorm};

let product = LogicFamily::Product;
let and_degree = product.tnorm(0.5, 0.6);
let or_degree = product.tconorm(0.5, 0.6);

assert!((or_degree - 0.8).abs() < 1e-12);
assert!((and_degree - 0.3).abs() < 1e-12);

// The same conjunction with the logic as a compile-time type parameter.
assert_eq!(<Product as TNorm>::tnorm(0.5, 0.6), and_degree);
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

## Which API to use

The three standard logics are reachable several ways, and every route
computes the same values:

| API | Use it when |
|-----|-------------|
| `TNorm` marker types (`<Product as TNorm>::tnorm(a, b)`) | The t-norm and residuum are the primary pair and the logic is a compile-time type parameter |
| `LogicFamily` (`Godel`, `Product`, `Lukasiewicz`) | The same pair at runtime, plus gradients, `f32` variants, or Burn tensors |
| `tconorm(family, a, b)`, folds | You need the full parametric catalog (`TConormFamily`: Frank, Yager, Dombi, ...) |
| `TruthAlgebra` marker types (`Godel`, `Product`, `Lukasiewicz`) | Type-level selection of the full algebra, including gradients and tensors |
| `tnorm(family, a, b)` (deprecated) | Legacy code only; use `TNorm` or `LogicFamily` instead |

`TConormFamily` is an aggregation catalog named after its t-conorm. Each
variant also exposes the dual t-norm `T(a, b) = 1 - S(1 - a, 1 - b)`, but that
route is deprecated: the t-norm and its residuum are the primary pair, so
prefer `TNorm` or `LogicFamily`. `PRODUCT` is the `Probabilistic` variant, so
`tconorm(PRODUCT, a, b)` is `a + b - a * b` and the deprecated
`tnorm(PRODUCT, a, b)` is still `a * b`.

Residual negation is `¬a = a → 0` (`TNorm::negation`, `LogicFamily::neg`).
It is involutive only for Lukasiewicz; `Godel` and `Product` give a two-valued
negation, so `¬¬a = a` does not hold. That is why each built-in family overrides
`tconorm` with the standard-negation dual: the literal duality
`a ⊕ b = ¬(¬a ⊗ ¬b)` collapses to the constant `1` under a non-involutive
negation.

Inputs outside `[0, 1]` are clamped to `[0, 1]` by every family. `tnorm` and
`tconorm` treat NaN as 0; the residuum, gradient and fold functions can return
NaN for NaN input.

## Examples

```sh
cargo run --example logic_table
cargo run --example aggregation_recipe
```
