# tnorms

[![crates.io](https://img.shields.io/crates/v/tnorms.svg)](https://crates.io/crates/tnorms)
[![Documentation](https://docs.rs/tnorms/badge.svg)](https://docs.rs/tnorms)
[![CI](https://github.com/arclabs561/tnorms/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/tnorms/actions/workflows/ci.yml)

T-norm and t-conorm families.

`tnorms` provides common fuzzy-logic aggregation families and fold helpers for
multi-way conjunction and disjunction over values in `[0, 1]`.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/tnorms) | [docs.rs](https://docs.rs/tnorms)

```toml
[dependencies]
tnorms = "0.1.1"
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
