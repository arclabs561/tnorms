use crate::{tconorm, tnorm, LogicFamily, TConormFamily, FRANK_PRODUCT_LIMIT_EPS};
#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Fuzzy negation family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NegationFamily {
    /// Standard involutive negation `1 - x`.
    Standard,
    /// Sugeno negation `(1 - x) / (1 + lambda*x)`, `lambda > -1`.
    Sugeno {
        /// Shape parameter.
        lambda: f64,
    },
    /// Yager negation `(1 - x^p)^(1/p)`, `p > 0`.
    Yager {
        /// Shape parameter.
        p: f64,
    },
}

/// Evaluate a fuzzy negation.
pub fn negation(family: NegationFamily, x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    match family {
        NegationFamily::Standard => 1.0 - x,
        NegationFamily::Sugeno { lambda } => {
            debug_assert!(lambda > -1.0, "Sugeno lambda must be > -1");
            ((1.0 - x) / lambda.mul_add(x, 1.0)).clamp(0.0, 1.0)
        }
        NegationFamily::Yager { p } => {
            debug_assert!(p > 0.0, "Yager p must be > 0");
            (1.0 - x.powf(p)).max(0.0).powf(1.0 / p)
        }
    }
}

/// Fuzzy implication family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplicationFamily {
    /// Kleene-Dienes implication `max(1-a, b)`.
    KleeneDienes,
    /// Reichenbach implication `1 - a + a*b`.
    Reichenbach,
    /// Lukasiewicz implication `min(1, 1-a+b)`.
    Lukasiewicz,
    /// Godel implication.
    Godel,
    /// Goguen/Product implication.
    Product,
}

/// Evaluate a fuzzy implication `a -> b`.
pub fn implication(family: ImplicationFamily, a: f64, b: f64) -> f64 {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    match family {
        ImplicationFamily::KleeneDienes => (1.0 - a).max(b),
        ImplicationFamily::Reichenbach => 1.0 - a + a * b,
        ImplicationFamily::Lukasiewicz => (1.0 - a + b).min(1.0),
        ImplicationFamily::Godel => LogicFamily::Godel.residuum(a, b),
        ImplicationFamily::Product => LogicFamily::Product.residuum(a, b),
    }
}

/// Ordered weighted averaging over `values`.
///
/// `values` is sorted in place from largest to smallest. Returns `None` when
/// the slices differ in length, are empty, or the weights do not sum to `1`.
pub fn owa(weights: &[f64], values: &mut [f64]) -> Option<f64> {
    if weights.is_empty() || weights.len() != values.len() {
        return None;
    }
    let weight_sum: f64 = weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-9 {
        return None;
    }

    for i in 0..values.len() {
        let mut max_index = i;
        for j in (i + 1)..values.len() {
            if values[j] > values[max_index] {
                max_index = j;
            }
        }
        values.swap(i, max_index);
    }
    Some(
        weights
            .iter()
            .zip(values.iter())
            .map(|(weight, value)| weight * value.clamp(0.0, 1.0))
            .sum::<f64>()
            .clamp(0.0, 1.0),
    )
}

/// Generalized power mean over values in `[0, 1]`.
pub fn power_mean(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let weight = 1.0 / values.len() as f64;
    power_mean_weighted(values, p, |_| weight)
}

/// Weighted generalized power mean over values in `[0, 1]`.
pub fn power_mean_weighted<F>(values: &[f64], p: f64, weight: F) -> Option<f64>
where
    F: Fn(usize) -> f64,
{
    if values.is_empty() {
        return None;
    }
    let mut weight_sum = 0.0;
    if p == f64::INFINITY {
        return Some(
            values
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.clamp(0.0, 1.0))),
        );
    }
    if p == f64::NEG_INFINITY {
        return Some(
            values
                .iter()
                .fold(1.0_f64, |acc, value| acc.min(value.clamp(0.0, 1.0))),
        );
    }
    if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
        let mut log_sum = 0.0;
        for (index, value) in values.iter().enumerate() {
            let weight = weight(index);
            weight_sum += weight;
            log_sum += weight * value.clamp(1e-300, 1.0).ln();
        }
        return if (weight_sum - 1.0).abs() <= 1e-9 {
            Some(log_sum.exp())
        } else {
            None
        };
    }

    let mut sum = 0.0;
    for (index, value) in values.iter().enumerate() {
        let weight = weight(index);
        weight_sum += weight;
        sum += weight * value.clamp(0.0, 1.0).powf(p);
    }
    if (weight_sum - 1.0).abs() <= 1e-9 {
        Some(sum.max(0.0).powf(1.0 / p).clamp(0.0, 1.0))
    } else {
        None
    }
}

/// Sum of logs for product t-norm aggregation.
///
/// This computes `sum(ln(max(value, floor)))`, the log-space equivalent of
/// multiplying all clamped values. The empty product returns `0.0`, since
/// `ln(1) = 0`.
pub fn log_product(values: &[f64], floor: f64) -> f64 {
    let floor = if floor.is_finite() && floor > 0.0 {
        floor.min(1.0)
    } else {
        f64::MIN_POSITIVE
    };
    values
        .iter()
        .map(|value| value.clamp(floor, 1.0).ln())
        .sum()
}

/// Convert a log-space product back to a degree in `[0, 1]`.
pub fn product_from_log(log_product: f64) -> f64 {
    if log_product.is_nan() || log_product == f64::NEG_INFINITY {
        0.0
    } else if log_product == f64::INFINITY {
        1.0
    } else {
        log_product.exp().clamp(0.0, 1.0)
    }
}

/// Mixed-region behavior for representable uninorms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedRegion {
    /// Use `min(a, b)` when one input is below the neutral point and the other is above.
    Min,
    /// Use `max(a, b)` when one input is below the neutral point and the other is above.
    Max,
}

/// Evaluate a representable uninorm with neutral point `e`.
///
/// Below `e`, inputs are rescaled into `[0, 1]` and combined with `tnorm_family`.
/// Above `e`, inputs are rescaled and combined with `tconorm_family`.
pub fn representable_uninorm(
    tnorm_family: TConormFamily,
    tconorm_family: TConormFamily,
    mixed: MixedRegion,
    e: f64,
    a: f64,
    b: f64,
) -> f64 {
    let e = e.clamp(0.0, 1.0);
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    if e <= 0.0 {
        return tconorm(tconorm_family, a, b);
    }
    if e >= 1.0 {
        return tnorm(tnorm_family, a, b);
    }

    if a <= e && b <= e {
        e * tnorm(tnorm_family, a / e, b / e)
    } else if a >= e && b >= e {
        e + (1.0 - e) * tconorm(tconorm_family, (a - e) / (1.0 - e), (b - e) / (1.0 - e))
    } else {
        match mixed {
            MixedRegion::Min => a.min(b),
            MixedRegion::Max => a.max(b),
        }
    }
}
