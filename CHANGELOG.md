# Changelog

## [Unreleased]

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
