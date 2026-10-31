use num_traits::Float;

use crate::{tnorm_gradient, BinaryGradient, TConormFamily};

pub(crate) const FRANK_MAXIMUM_LIMIT_LOG: f64 = -350.0;
pub(crate) const FRANK_PRODUCT_LIMIT_EPS: f64 = 1e-7;
pub(crate) const FRANK_BOUNDED_LIMIT_LOG: f64 = 350.0;

pub(crate) fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::nan)
}

pub(crate) fn clamp01<T: Float>(value: T) -> T {
    value.max(T::zero()).min(T::one())
}

/// Clamp a computed result to `[0, 1]` without turning NaN into a number, so
/// rounding cannot push an output outside the unit interval.
pub(crate) fn clamp_result<T: Float>(value: T) -> T {
    if value > T::one() {
        T::one()
    } else if value < T::zero() {
        T::zero()
    } else {
        value
    }
}

pub(crate) fn scaled_lp_norm_2<T: Float>(a: T, b: T, p: T) -> T {
    let scale = a.max(b);
    if scale <= T::zero() {
        return T::zero();
    }
    let scaled_sum = (a / scale).powf(p) + (b / scale).powf(p);
    scale * scaled_sum.powf(T::one() / p)
}

pub(crate) fn frank_tconorm<T: Float>(a: T, b: T, s: T) -> T {
    if (s - T::one()).abs() <= cast(FRANK_PRODUCT_LIMIT_EPS) {
        return a + b - a * b;
    }

    let log_s = s.ln();
    if log_s <= cast(FRANK_MAXIMUM_LIMIT_LOG) {
        return a.max(b);
    }
    if log_s >= cast(FRANK_BOUNDED_LIMIT_LOG) {
        return (a + b).min(T::one());
    }

    if log_s < -T::one() {
        return frank_tconorm_small_base(a, b, log_s);
    }

    let denominator = log_s.exp_m1();
    let left = ((T::one() - a) * log_s).exp_m1();
    let right = ((T::one() - b) * log_s).exp_m1();

    clamp01(T::one() - (left * right / denominator).ln_1p() / log_s)
}

/// Frank t-conorm for `ln s < 0`, written so nothing cancels as `s -> 0`.
///
/// With `L = -ln s`, `x = 1-a`, `y = 1-b`, the log argument of
/// `S = 1 - ln(1 + (s^x-1)(s^y-1)/(s-1)) / ln s` is
/// `((s^x - s) + s^y (1 - s^x)) / (1 - s)`: a sum of nonnegative terms, taken
/// in log space. The direct form rounds `1 + (-1)` to zero once `s^x`, `s^y`
/// fall below machine epsilon.
fn frank_tconorm_small_base<T: Float>(a: T, b: T, log_s: T) -> T {
    let l = -log_s;
    let one = T::one();
    // ln(s^x - s) = -x*L + ln(1 - s^(1-x)), with 1-x = a.
    let t1 = -(one - a) * l + (-(-a * l).exp_m1()).ln();
    // ln(s^y (1 - s^x)).
    let t2 = -(one - b) * l + (-(-(one - a) * l).exp_m1()).ln();
    let sum = if t1 == T::neg_infinity() {
        t2
    } else if t2 == T::neg_infinity() {
        t1
    } else {
        log_sum_exp_2(t1, t2)
    };
    let ln_inner = sum - (-log_s.exp()).ln_1p();
    clamp01(one - ln_inner / log_s)
}

pub(crate) fn dombi_tconorm<T: Float>(a: T, b: T, p: T) -> T {
    if a <= T::zero() {
        return b;
    }
    if b <= T::zero() {
        return a;
    }
    if a >= T::one() || b >= T::one() {
        return T::one();
    }

    let log_ta = p * ((T::one() - a).ln() - a.ln());
    let log_tb = p * ((T::one() - b).ln() - b.ln());
    let log_inner = -log_sum_exp_2(-log_ta, -log_tb);
    let inner_root = (log_inner / p).exp();
    clamp01(T::one() / (T::one() + inner_root))
}

pub(crate) fn log_sum_exp_2<T: Float>(a: T, b: T) -> T {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Hamacher product `T(a, b) = ab / (a + b - ab)`.
///
/// The denominator is written as `x + y (1 - x)` with `x = max(a, b)` and
/// `y = min(a, b)`: a sum of nonnegative terms, so nothing cancels, and it is
/// exactly 1 when `x = 1`. That makes `T(y, 1) = y` and `T(y, 0) = 0` exact.
/// Computing `T` as the dual `1 - S(1-a, 1-b)` cancelled near `a = 1`.
pub(crate) fn hamacher_tnorm<T: Float>(a: T, b: T) -> T {
    let (x, y) = if a >= b { (a, b) } else { (b, a) };
    let denominator = x + y * (T::one() - x);
    if denominator <= T::zero() {
        return T::zero();
    }
    clamp_result(a * b / denominator)
}

/// Hamacher t-conorm `S(a, b) = (a + b - 2ab) / (1 - ab)`, written as
/// `S = x + y (1-x)^2 / (1 - xy)` with `x = max(a, b)` and `y = min(a, b)`.
///
/// The correction term is at most `1 - x`, so the cancellation in `1 - xy`
/// near `(1, 1)` costs no more than an ulp of the result, and the identities
/// `S(y, 0) = y` and `S(y, 1) = 1` are exact. The denominator is formed as
/// `(1 - y) + y (1 - x)`, which does not cancel.
pub(crate) fn hamacher_tconorm<T: Float>(a: T, b: T) -> T {
    let (x, y) = if a >= b { (a, b) } else { (b, a) };
    let gap = T::one() - x;
    let denominator = (T::one() - y) + y * gap;
    if denominator <= T::zero() {
        // x = y = 1.
        return T::one();
    }
    clamp_result(x + y * gap * gap / denominator)
}

/// `ln(1 - e^t)` for `t < 0` without cancellation: `ln_1p(-e^t)` when `e^t`
/// is small and `ln(-expm1(t))` when it is close to 1 (the `log1mexp`
/// split of Maechler, "Accurately Computing log(1 - exp(-|a|))", 2012).
pub(crate) fn ln_one_minus_exp(t: f64) -> f64 {
    if t < -core::f64::consts::LN_2 {
        (-t.exp()).ln_1p()
    } else {
        (-t.exp_m1()).ln()
    }
}

pub(crate) fn drastic_tnorm<T: Float>(a: T, b: T) -> T {
    if a >= T::one() {
        b
    } else if b >= T::one() {
        a
    } else {
        T::zero()
    }
}

pub(crate) fn drastic_tconorm<T: Float>(a: T, b: T) -> T {
    if a <= T::zero() {
        b
    } else if b <= T::zero() {
        a
    } else {
        T::one()
    }
}

pub(crate) fn nilpotent_minimum_tnorm<T: Float>(a: T, b: T) -> T {
    if a + b > T::one() {
        a.min(b)
    } else {
        T::zero()
    }
}

pub(crate) fn nilpotent_minimum_tconorm<T: Float>(a: T, b: T) -> T {
    if a + b < T::one() {
        a.max(b)
    } else {
        T::one()
    }
}

pub(crate) fn schweizer_sklar_tnorm<T: Float>(a: T, b: T, p: T) -> T {
    if p == T::neg_infinity() {
        return a.min(b);
    }
    if p == T::infinity() {
        return drastic_tnorm(a, b);
    }
    if p.abs() <= cast(FRANK_PRODUCT_LIMIT_EPS) {
        return a * b;
    }
    let (x, y) = if a >= b { (a, b) } else { (b, a) };
    if y <= T::zero() {
        // p > 0: (x^p - 1)^(1/p) clamps to 0; p < 0: the zero is absorbing.
        return T::zero();
    }

    // T = (x^p + y^p - 1)^(1/p) = y (1 + (x^p - 1) / y^p)^(1/p), with
    // x = max(a, b), y = min(a, b). Forming x^p + y^p - 1 directly loses
    // y^p next to 1 (for p = 20, T(0.05, 1) came out as 0). Here x^p - 1 is
    // expm1(p ln x), exactly 0 at x = 1, so T(y, 1) = y.
    let x_power_minus_one = (p * x.ln()).exp_m1();
    if x_power_minus_one == T::zero() {
        return y;
    }
    let ratio = if x_power_minus_one.is_finite() {
        x_power_minus_one * (-p * y.ln()).exp()
    } else {
        // p < 0 and x^p overflows: (x^p - 1) / y^p = (x/y)^p - y^-p, and
        // y^-p < x^-p underflows, since y <= x.
        (p * (x.ln() - y.ln())).exp()
    };
    if ratio <= -T::one() {
        return T::zero();
    }
    clamp_result(y * (ratio.ln_1p() / p).exp())
}

pub(crate) fn sugeno_weber_tnorm<T: Float>(a: T, b: T, p: T) -> T {
    if p <= -T::one() {
        return drastic_tnorm(a, b);
    }
    if p == T::infinity() {
        return a * b;
    }
    let numerator = a + b - T::one() + p * a * b;
    clamp01((numerator / (T::one() + p)).max(T::zero()))
}

pub(crate) fn aczel_alsina_tnorm<T: Float>(a: T, b: T, p: T) -> T {
    if p <= T::zero() {
        return drastic_tnorm(a, b);
    }
    if p == T::infinity() {
        return a.min(b);
    }
    if a <= T::zero() || b <= T::zero() {
        return T::zero();
    }
    if a >= T::one() {
        return b;
    }
    if b >= T::one() {
        return a;
    }

    let left = -a.ln();
    let right = -b.ln();
    (-scaled_lp_norm_2(left, right, p)).exp()
}

pub(crate) fn mayor_torrens_tnorm<T: Float>(a: T, b: T, p: T) -> T {
    let p = clamp01(p);
    if p <= T::zero() {
        return a.min(b);
    }
    if p >= T::one() {
        return (a + b - T::one()).max(T::zero());
    }
    if a <= p && b <= p {
        (a + b - p).max(T::zero())
    } else {
        a.min(b)
    }
}

pub(crate) fn hamacher_generator(x: f64, p: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if p <= 0.0 {
        (1.0 - x) / x
    } else {
        ((p + (1.0 - p) * x) / x).ln()
    }
}

pub(crate) fn hamacher_generator_inverse(y: f64, p: f64) -> f64 {
    if p <= 0.0 {
        1.0 / (1.0 + y)
    } else {
        p / (y.exp() - 1.0 + p)
    }
}

/// Frank generator `f(x) = ln((s - 1) / (s^x - 1))`.
///
/// For `s < 1` this is `ln(1 - s) - ln(1 - s^x)`, with each term from
/// [`ln_one_minus_exp`]. The quotient form rounds to `ln(1) = 0` once `s^x`
/// falls below machine epsilon, which loses the whole value.
pub(crate) fn frank_generator(x: f64, s: f64) -> f64 {
    if (s - 1.0).abs() <= FRANK_PRODUCT_LIMIT_EPS {
        return crate::GeneratorFamily::Product.value(x);
    }
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let log_s = s.ln();
    if log_s < 0.0 {
        (ln_one_minus_exp(log_s) - ln_one_minus_exp(x * log_s)).max(0.0)
    } else {
        (log_s.exp_m1().ln() - (x * log_s).exp_m1().ln()).max(0.0)
    }
}

/// Frank generator inverse `f^-1(y) = ln(1 + (s - 1) e^-y) / ln s`.
///
/// For `s < 1` the log argument `1 - (1 - s) e^-y` is written as
/// `(1 - e^-y) + s e^-y`, a sum of nonnegative terms, when `(1 - s) e^-y`
/// is close to 1, and taken through `ln_1p` when it is small; the direct
/// form rounds `1 + (s - 1)` to 0 for small `s` and `y` and returns
/// `ln(0) / ln s = +inf`. For `s > 1` it is `ln_1p((s - 1) e^-y)`.
pub(crate) fn frank_generator_inverse(y: f64, s: f64) -> f64 {
    if (s - 1.0).abs() <= FRANK_PRODUCT_LIMIT_EPS {
        return crate::GeneratorFamily::Product.inverse(y);
    }
    let log_s = s.ln();
    let value = if log_s < 0.0 {
        let shrink = -log_s.exp_m1() * (-y).exp(); // (1 - s) e^-y
        if shrink < 0.5 {
            (-shrink).ln_1p() / log_s
        } else {
            (-(-y).exp_m1() + (log_s - y).exp()).ln() / log_s
        }
    } else {
        (log_s.exp_m1() * (-y).exp()).ln_1p() / log_s
    };
    value.clamp(0.0, 1.0)
}

pub(crate) fn min_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
    if a < b {
        BinaryGradient::new(1.0, 0.0)
    } else if a > b {
        BinaryGradient::new(0.0, 1.0)
    } else {
        BinaryGradient::new(0.5, 0.5)
    }
}

pub(crate) fn max_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
    if a > b {
        BinaryGradient::new(1.0, 0.0)
    } else if a < b {
        BinaryGradient::new(0.0, 1.0)
    } else {
        BinaryGradient::new(0.5, 0.5)
    }
}

pub(crate) fn hamacher_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if a <= 0.0 || b <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let denom = p + (1.0 - p) * (a + b - a * b);
    if denom.abs() < 1e-15 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let denom_sq = denom * denom;
    BinaryGradient::new(
        b * (p + (1.0 - p) * b) / denom_sq,
        a * (p + (1.0 - p) * a) / denom_sq,
    )
}

pub(crate) fn frank_tnorm_gradient(a: f64, b: f64, s: f64) -> BinaryGradient<f64> {
    if (s - 1.0).abs() <= FRANK_PRODUCT_LIMIT_EPS {
        return BinaryGradient::new(b, a);
    }
    let log_s = s.ln();
    if log_s <= FRANK_MAXIMUM_LIMIT_LOG {
        return min_gradient(a, b);
    }
    if log_s >= FRANK_BOUNDED_LIMIT_LOG {
        return tnorm_gradient(TConormFamily::Bounded, a, b);
    }
    let sx = (a * log_s).exp();
    let sy = (b * log_s).exp();
    let denom = s - 1.0 + (sx - 1.0) * (sy - 1.0);
    if denom.abs() < 1e-15 {
        return BinaryGradient::new(0.0, 0.0);
    }
    BinaryGradient::new(sx * (sy - 1.0) / denom, sy * (sx - 1.0) / denom)
}

pub(crate) fn yager_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if p <= 0.0 {
        return drastic_tnorm_gradient(a, b);
    }
    let left = 1.0 - a;
    let right = 1.0 - b;
    let inner = left.powf(p) + right.powf(p);
    if inner >= 1.0 || inner <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let scale = inner.powf(1.0 / p - 1.0);
    BinaryGradient::new(left.powf(p - 1.0) * scale, right.powf(p - 1.0) * scale)
}

pub(crate) fn dombi_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if a <= 0.0 || b <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    if a >= 1.0 {
        return BinaryGradient::new(0.0, 1.0);
    }
    if b >= 1.0 {
        return BinaryGradient::new(1.0, 0.0);
    }

    let left = (1.0 - a) / a;
    let right = (1.0 - b) / b;
    let width = scaled_lp_norm_2(left, right, p);
    if width <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let value = 1.0 / (1.0 + width);
    let common = value * value * width.powf(1.0 - p);
    BinaryGradient::new(
        common * left.powf(p - 1.0) / (a * a),
        common * right.powf(p - 1.0) / (b * b),
    )
}

pub(crate) fn drastic_tnorm_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
    if a >= 1.0 {
        BinaryGradient::new(0.0, 1.0)
    } else if b >= 1.0 {
        BinaryGradient::new(1.0, 0.0)
    } else {
        BinaryGradient::new(0.0, 0.0)
    }
}

pub(crate) fn nilpotent_minimum_tnorm_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
    if a + b > 1.0 {
        min_gradient(a, b)
    } else {
        BinaryGradient::new(0.0, 0.0)
    }
}

pub(crate) fn schweizer_sklar_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if p == f64::NEG_INFINITY {
        return min_gradient(a, b);
    }
    if p == f64::INFINITY {
        return drastic_tnorm_gradient(a, b);
    }
    if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
        return BinaryGradient::new(b, a);
    }
    let value = schweizer_sklar_tnorm(a, b, p);
    if value <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    BinaryGradient::new(
        a.powf(p - 1.0) * value.powf(1.0 - p),
        b.powf(p - 1.0) * value.powf(1.0 - p),
    )
}

pub(crate) fn sugeno_weber_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if p <= -1.0 {
        return drastic_tnorm_gradient(a, b);
    }
    if p == f64::INFINITY {
        return BinaryGradient::new(b, a);
    }
    if a + b - 1.0 + p * a * b <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    BinaryGradient::new((1.0 + p * b) / (1.0 + p), (1.0 + p * a) / (1.0 + p))
}

pub(crate) fn aczel_alsina_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    if p <= 0.0 {
        return drastic_tnorm_gradient(a, b);
    }
    if p == f64::INFINITY {
        return min_gradient(a, b);
    }
    if a <= 0.0 || b <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let left = -a.ln();
    let right = -b.ln();
    let width = scaled_lp_norm_2(left, right, p);
    if width <= 0.0 {
        return BinaryGradient::new(0.0, 0.0);
    }
    let value = (-width).exp();
    BinaryGradient::new(
        value * (left / width).powf(p - 1.0) / a,
        value * (right / width).powf(p - 1.0) / b,
    )
}

pub(crate) fn mayor_torrens_tnorm_gradient(a: f64, b: f64, p: f64) -> BinaryGradient<f64> {
    let p = p.clamp(0.0, 1.0);
    if p <= 0.0 {
        return min_gradient(a, b);
    }
    if p >= 1.0 {
        return tnorm_gradient(TConormFamily::Bounded, a, b);
    }
    if a <= p && b <= p {
        if a + b > p {
            BinaryGradient::new(1.0, 1.0)
        } else {
            BinaryGradient::new(0.0, 0.0)
        }
    } else {
        min_gradient(a, b)
    }
}
