# Changelog

## [Unreleased]

### Added

- `TNorm`, the primary t-norm surface: `NAME`, `tnorm`, `residuum`,
  `negation`, and a provided `tconorm` derived by duality. It is implemented
  for the `Godel`, `Product` and `Lukasiewicz` marker types and delegates to
  `LogicFamily`, so every value is bit-identical.
- Adjunction tests (`a ⊗ b <= c` iff `a <= b → c`, tolerance `1e-12`) over a
  grid for the three logics, plus a negation-involution test that only
  Łukasiewicz satisfies.

### Deprecated

- `tnorm(TConormFamily, ..)`, `tnorm_generic`, and the
  `TConormFamily::tnorm`/`TConormFamily::tnorm_generic` methods. The t-norm is
  not defined by its t-conorm; use `TNorm` or `LogicFamily::tnorm`. Values are
  unchanged and the items are scheduled for removal in 0.3.0.

### Fixed

- Hamacher's t-norm uses the Hamacher product `ab / (a + b - ab)` directly,
  and its t-conorm is computed as `x + y (1-x)^2 / (1 - xy)` with
  `x = max(a, b)`, `y = min(a, b)`. The t-norm was the dual of
  `(a + b - 2ab) / (1 - ab)`, which cancels near 1: `T(1e-12, 0)` was
  `1.1e-4` and `S(1 - 1e-12, 1)` was `0.99989`. The identities `T(x, 1) = x`,
  `T(x, 0) = 0`, `S(x, 0) = x` and `S(x, 1) = 1` are now exact.
- The Schweizer-Sklar t-norm forms `x^p - 1` with `expm1`, so `T(x, 1) = x`
  holds exactly for every `p`. `(x^p + 1) - 1` lost `x^p` for large `p`:
  `T(0.05, 1)` was 0 at `p = 20`.
- `GeneratorFamily::residuum` (and `residuum` for Archimedean families) no
  longer returns 1 when the generator overflows. `f(b) - f(a)` was
  `inf - inf = NaN`, so Dombi with `p = 50` gave `residuum(1e-12, 0) = 1`
  instead of 0. A zero consequent of a strict generator now gives 0, and
  overflowing Dombi, Aczel-Alsina and negative-`p` Schweizer-Sklar values
  are subtracted in log space.
- The Frank generator and its inverse stay accurate for small `s`. The
  inverse computed `ln(1 + (s - 1) e^-y)`, which reached `ln(0)` and
  returned `+inf` for `GeneratorFamily::Frank { s: 1e-200 }.tnorm(0.1, 0.1)`.

## [0.2.1] - 2026-10-09

### Fixed

- The Frank t-conorm uses a stable form when `ln s < -1`. The direct form
  cancelled to 0 once `s^(1-a)` fell below machine epsilon (for example
  0.2227 at `s = 1e-20` and 0 for smaller `s`).
- The Hamacher t-conorm and t-norms computed as the dual of a t-conorm clamp
  their results to `[0, 1]`. Rounding previously gave `1.0000000000000002`
  for Hamacher at `a = 1` and `-2.2e-16` for a dual t-norm at `a = 0`. NaN
  inputs still produce NaN.
- `residuum` for `MayorTorrens` uses its closed form instead of a numeric
  supremum. Parameter limits dispatch to the matching basic logic:
  `SchweizerSklar` at `p = -inf`, `AczelAlsina` and `Yager` at `p = inf`,
  and Frank with `ln s <= -350` give the Godel residuum; `SugenoWeber` at
  `p = inf` gives the Product residuum; Frank with `ln s >= 350` gives the
  Lukasiewicz residuum.

## [0.2.0] - 2026-07-08

### Added

- Added `TruthAlgebra` marker types for Godel, Product, and Lukasiewicz logics.
- Added native `f32` implementations for `LogicFamily`.
- Added scalar gradients for Godel, Product, and Lukasiewicz logic operators.
- Added optional Burn tensor operations for the core logic families.
- Added generic-float t-norm and t-conorm evaluation.
- Added Schweizer-Sklar, Sugeno-Weber, Aczel-Alsina, Mayor-Torrens, Drastic,
  and Nilpotent-minimum catalog families.
- Added additive-generator, catalog residuum, and catalog gradient APIs.
- Added conservative copula-compatibility metadata for catalog families.
- Added borrowed-slice ordinal sums over disjoint sub-intervals.
- Added fuzzy negation, implication, OWA, power-mean, and representable-uninorm
  operators.
- Added `log_product` and `product_from_log` helpers for stable product
  aggregation in log space.
- Added a no-std scalar build path.

### Changed

- Hardened Frank, Yager, and Dombi t-conorm numerics near singular limits and
  high exponents.

## [0.1.2] - 2026-07-07

### Added

- Added `LogicFamily` with t-norm, t-conorm, residuum, negation, and `f32`
  helpers for Godel, Product, and Lukasiewicz logics.

## [0.1.1] - 2026-07-07

### Added

- Added `GODEL`, `PRODUCT`, and `LUKASIEWICZ` named family constants.

## [0.1.0] - 2026-07-07

### Added

- Initial t-norm and t-conorm family catalog.
