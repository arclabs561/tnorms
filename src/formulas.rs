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

    let denominator = log_s.exp_m1();
    let left = ((T::one() - a) * log_s).exp_m1();
    let right = ((T::one() - b) * log_s).exp_m1();

    clamp01(T::one() - (left * right / denominator).ln_1p() / log_s)
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
    if p < T::zero() && (a <= T::zero() || b <= T::zero()) {
        return T::zero();
    }

    let ap = (p * a.ln()).exp();
    let bp = (p * b.ln()).exp();
    let inner = if p > T::zero() {
        (ap + bp - T::one()).max(T::zero())
    } else {
        ap + bp - T::one()
    };
    if inner <= T::zero() {
        T::zero()
    } else {
        clamp01(inner.powf(T::one() / p))
    }
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
    ((s - 1.0) / (x * s.ln()).exp_m1()).ln()
}

pub(crate) fn frank_generator_inverse(y: f64, s: f64) -> f64 {
    if (s - 1.0).abs() <= FRANK_PRODUCT_LIMIT_EPS {
        return crate::GeneratorFamily::Product.inverse(y);
    }
    (1.0 + (s - 1.0) * (-y).exp()).ln() / s.ln()
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
