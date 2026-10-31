//! T-conorm (s-norm) families for fuzzy logic and differentiable relaxations.
//!
//! A t-conorm `S: [0, 1]^2 -> [0, 1]` generalizes logical OR. It is:
//! - Commutative: S(a, b) = S(b, a)
//! - Associative: S(S(a, b), c) = S(a, S(b, c))
//! - Monotone: a1 <= a2 => S(a1, b) <= S(a2, b)
//! - Bounded: S(a, 0) = a (0 is identity)
//!
//! Different families provide different "softness" of the OR operation,
//! useful in differentiable relaxations and probabilistic reasoning.
//!
//! ## Primary surface
//!
//! Fuzzy logic is built on a left-continuous t-norm and its residuum; the
//! t-conorm is their dual under a strong negation. [`TNorm`] is that primary
//! surface for the three standard logics, and [`LogicFamily`] is the runtime
//! equivalent:
//!
//! ```
//! use tnorms::{LogicFamily, Lukasiewicz, TNorm};
//!
//! assert!((<Lukasiewicz as TNorm>::tnorm(0.4, 0.7) - 0.1).abs() < 1e-12);
//! assert!((<Lukasiewicz as TNorm>::residuum(0.7, 0.4) - 0.7).abs() < 1e-12);
//! assert!((<Lukasiewicz as TNorm>::negation(0.4) - 0.6).abs() < 1e-12);
//! assert_eq!(
//!     <Lukasiewicz as TNorm>::tconorm(0.4, 0.7),
//!     LogicFamily::Lukasiewicz.tconorm(0.4, 0.7)
//! );
//! ```
//!
//! ## Named families
//!
//! The constants [`GODEL`], [`PRODUCT`], and [`LUKASIEWICZ`] name the three
//! common fuzzy-logic families. [`LogicFamily`] adds the corresponding
//! residuum for implication:
//!
//! ```
//! assert_eq!(tnorms::LogicFamily::Godel.tnorm(0.4, 0.7), 0.4);
//! assert!((tnorms::LogicFamily::Product.tnorm(0.4, 0.7) - 0.28).abs() < 1e-12);
//! assert!((<tnorms::Lukasiewicz as tnorms::TNorm>::tnorm(0.4, 0.7) - 0.1).abs() < 1e-12);
//! assert_eq!(tnorms::LogicFamily::Godel.residuum(0.7, 0.4), 0.4);
//! assert_eq!(tnorms::LogicFamily::Product.tnorm_gradient(0.4, 0.7).da, 0.7);
//! ```
//!
//! [`TruthAlgebra`] exposes the same three families as zero-sized marker types
//! for crates that choose the logic at the type level:
//!
//! ```
//! use tnorms::{Product, TruthAlgebra};
//!
//! assert!((Product::tnorm_f32(0.4, 0.7) - 0.28).abs() < 1e-6);
//! ```
//!
//! The generic [`TConormFamily`] catalog is an aggregation catalog: it names
//! t-conorm formulas and derives a dual t-norm where useful. Its `tnorm`
//! functions are deprecated in favour of [`TNorm`]; [`residuum`] remains
//! available for catalog families with a known left-continuous path.
//! [`LogicFamily`] remains the runtime fuzzy-logic surface where the t-norm
//! and residuum form the adjoint pair by construction.
//!
//! ## Catalog
//!
//! | Family | Formula S(a,b) | Parameter | Behavior |
//! |--------|---------------|-----------|----------|
//! | Maximum | max(a,b) | none | Hard OR |
//! | Probabilistic | a+b-ab | none | Independent events |
//! | Bounded (Lukasiewicz) | min(1, a+b) | none | Saturating sum |
//! | Einstein | (a+b)/(1+ab) | none | Smooth OR |
//! | Hamacher | (a+b-2ab)/(1-ab) | none | Aggressive |
//! | Yager | min(1, (a^p + b^p)^(1/p)) | p >= 1 | Lp norm |
//! | Frank | 1 - log_s(1+(s^(1-a)-1)(s^(1-b)-1)/(s-1)) | s > 0 | Interpolates max<->bounded |
//! | Dombi | 1/(1+((1/a-1)^p + (1/b-1)^p)^(-1/p)) | p > 0 | Power-based |
//! | Schweizer-Sklar | dual of `(x^p+y^p-1)^(1/p)` | p | spans min/product/Łukasiewicz/drastic |
//! | Sugeno-Weber | dual of `max(0,(x+y-1+pxy)/(1+p))` | p >= -1 | spans drastic/Łukasiewicz/product |
//! | Aczel-Alsina | dual of `exp(-(((-ln x)^p+(-ln y)^p)^(1/p)))` | p > 0 | product-to-min family |
//! | Mayor-Torrens | ordinal-sum dual | `p in [0, 1]` | min-to-Łukasiewicz ordinal sum |
//! | Drastic | drastic sum | none | discontinuous limit |
//! | Nilpotent minimum | nilpotent-minimum dual | none | left-continuous threshold |
//!
//! Based on: Petersen et al., gendr -- Generalized Differentiable Rendering.
//! The t-conorm catalog provides the relaxation families used for differentiable
//! logical operations and soft aggregation.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

use num_traits::Float;

// Compile and run the README's Rust examples as doctests. One README example
// uses Burn tensors, so this runs under `--features burn` (CI: burn-ndarray).
#[cfg(all(doctest, feature = "burn"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod formulas;
mod operators;
mod ordinal;

use formulas::*;

pub use operators::{
    implication, log_product, negation, owa, power_mean, power_mean_weighted, product_from_log,
    representable_uninorm, ImplicationFamily, MixedRegion, NegationFamily,
};
pub use ordinal::{OrdinalSegment, OrdinalSum};

/// Evaluate a t-conorm S(a, b) for the given family.
pub fn tconorm(family: TConormFamily, a: f64, b: f64) -> f64 {
    tconorm_generic(family, a, b)
}

/// Evaluate a t-conorm S(a, b) for the given family and float type.
pub fn tconorm_generic<T: Float>(family: TConormFamily, a: T, b: T) -> T {
    let a = clamp01(a);
    let b = clamp01(b);

    match family {
        TConormFamily::Maximum => a.max(b),
        TConormFamily::Probabilistic => a + b - a * b,
        TConormFamily::Bounded => (a + b).min(T::one()),
        TConormFamily::Einstein => {
            let denom = T::one() + a * b;
            if denom == T::zero() {
                T::zero()
            } else {
                (a + b) / denom
            }
        }
        TConormFamily::Hamacher => hamacher_tconorm(a, b),
        TConormFamily::Yager { p } => {
            debug_assert!(p > 0.0, "Yager p must be > 0");
            scaled_lp_norm_2(a, b, cast(p)).min(T::one())
        }
        TConormFamily::Frank { s } => {
            debug_assert!(s > 0.0, "Frank s must be > 0");
            frank_tconorm(a, b, cast(s))
        }
        TConormFamily::Dombi { p } => {
            debug_assert!(p > 0.0, "Dombi p must be > 0");
            dombi_tconorm(a, b, cast(p))
        }
        TConormFamily::Drastic => drastic_tconorm(a, b),
        TConormFamily::NilpotentMinimum => nilpotent_minimum_tconorm(a, b),
        TConormFamily::SchweizerSklar { p } => {
            T::one() - schweizer_sklar_tnorm(T::one() - a, T::one() - b, cast(p))
        }
        TConormFamily::SugenoWeber { p } => {
            T::one() - sugeno_weber_tnorm(T::one() - a, T::one() - b, cast(p))
        }
        TConormFamily::AczelAlsina { p } => {
            T::one() - aczel_alsina_tnorm(T::one() - a, T::one() - b, cast(p))
        }
        TConormFamily::MayorTorrens { p } => {
            T::one() - mayor_torrens_tnorm(T::one() - a, T::one() - b, cast(p))
        }
    }
}

/// T-conorm family selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TConormFamily {
    /// S(a,b) = max(a,b). Hard OR, not differentiable at a=b.
    Maximum,
    /// S(a,b) = a+b-ab. Probabilistic sum (independent events).
    Probabilistic,
    /// S(a,b) = min(1, a+b). Lukasiewicz / bounded sum.
    Bounded,
    /// S(a,b) = (a+b)/(1+ab). Smooth, well-behaved for optimization.
    Einstein,
    /// S(a,b) = (a+b-2ab)/(1-ab). More aggressive than probabilistic.
    Hamacher,
    /// S(a,b) = min(1, (a^p + b^p)^(1/p)). Lp-norm family.
    Yager {
        /// Exponent, must be >= 1. p=1 gives Bounded, p->inf gives Maximum.
        p: f64,
    },
    /// Frank t-conorm parameterized by base s.
    /// s->0 gives Maximum, s=1 gives Probabilistic (limit), s->inf gives Bounded.
    Frank {
        /// Base parameter, must be > 0. `s=1` uses the probabilistic limit.
        s: f64,
    },
    /// Dombi t-conorm. p->0 gives Maximum, p->inf gives Bounded.
    Dombi {
        /// Exponent, must be > 0.
        p: f64,
    },
    /// Drastic sum, the dual of the drastic product.
    Drastic,
    /// Nilpotent-minimum dual t-conorm.
    NilpotentMinimum,
    /// Schweizer-Sklar t-conorm, parameterized by `p`.
    SchweizerSklar {
        /// Shape parameter. `p=0` is Product, `p=1` is Lukasiewicz,
        /// `p=-inf` is Godel, and `p=inf` is Drastic.
        p: f64,
    },
    /// Sugeno-Weber t-conorm, parameterized by the t-norm parameter `p`.
    SugenoWeber {
        /// Shape parameter, with valid finite values `p >= -1`.
        /// `p=-1` is Drastic, `p=0` is Lukasiewicz, and `p=inf` is Product.
        p: f64,
    },
    /// Aczel-Alsina t-conorm.
    AczelAlsina {
        /// Shape parameter, must be > 0. `p=1` is Product and `p->inf`
        /// approaches Godel.
        p: f64,
    },
    /// Mayor-Torrens ordinal-sum t-conorm.
    MayorTorrens {
        /// Łukasiewicz segment size in `[0, 1]`.
        p: f64,
    },
}

/// Whether a t-norm family is known to satisfy the copula constraints.
///
/// `ParameterDependent` and `Unknown` deliberately avoid collapsing uncertainty
/// to `false`: they mean callers should not use the family as a copula without
/// checking the chosen parameter range against a copula-specific reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopulaCompatibility {
    /// Every supported parameter value is a copula.
    Always,
    /// No supported parameter value is a copula.
    Never,
    /// Compatibility depends on the parameter value.
    ParameterDependent,
    /// Compatibility is not asserted by this crate.
    Unknown,
}

/// Short alias for selecting a t-norm/t-conorm family.
pub type Family = TConormFamily;

/// Godel family: `min(a, b)` t-norm and `max(a, b)` t-conorm.
pub const GODEL: Family = Family::Maximum;

/// Product family: `a * b` t-norm and `a + b - a * b` t-conorm.
pub const PRODUCT: Family = Family::Probabilistic;

/// Lukasiewicz family: `max(0, a + b - 1)` t-norm and `min(1, a + b)` t-conorm.
pub const LUKASIEWICZ: Family = Family::Bounded;

impl TConormFamily {
    /// Return conservative copula-compatibility metadata for this family.
    ///
    /// The three common logics (`Maximum`, `Probabilistic`, `Bounded`) are the
    /// Frechet-Hoeffding upper, independence, and lower copulas. Frank is the
    /// classical Frank copula family. Drastic and nilpotent-minimum violate the
    /// copula constraints. Other catalog families need parameter-specific
    /// treatment, so this method reports `ParameterDependent` or `Unknown`
    /// rather than pretending a t-norm law is also a copula law.
    pub fn copula_compatibility(self) -> CopulaCompatibility {
        match self {
            Self::Maximum | Self::Probabilistic | Self::Bounded | Self::Frank { .. } => {
                CopulaCompatibility::Always
            }
            Self::Drastic | Self::NilpotentMinimum => CopulaCompatibility::Never,
            Self::Dombi { .. }
            | Self::SchweizerSklar { .. }
            | Self::SugenoWeber { .. }
            | Self::AczelAlsina { .. }
            | Self::MayorTorrens { .. } => CopulaCompatibility::ParameterDependent,
            Self::Einstein | Self::Hamacher | Self::Yager { .. } => CopulaCompatibility::Unknown,
        }
    }

    /// Return `Some(true)` or `Some(false)` for families with unconditional
    /// copula status.
    ///
    /// Returns `None` when status depends on parameters or is not asserted.
    pub fn is_copula(self) -> Option<bool> {
        match self.copula_compatibility() {
            CopulaCompatibility::Always => Some(true),
            CopulaCompatibility::Never => Some(false),
            CopulaCompatibility::ParameterDependent | CopulaCompatibility::Unknown => None,
        }
    }

    /// Return this family's additive generator when it has one.
    ///
    /// Godel/minimum, drastic, nilpotent minimum, and Mayor-Torrens are not
    /// single Archimedean generator families on the whole unit interval.
    pub fn generator(self) -> Option<GeneratorFamily> {
        match self {
            Self::Maximum | Self::Drastic | Self::NilpotentMinimum | Self::MayorTorrens { .. } => {
                None
            }
            Self::Probabilistic => Some(GeneratorFamily::Product),
            Self::Bounded => Some(GeneratorFamily::Lukasiewicz),
            Self::Einstein => Some(GeneratorFamily::Hamacher { p: 2.0 }),
            Self::Hamacher => Some(GeneratorFamily::Hamacher { p: 0.0 }),
            Self::Yager { p } if p > 0.0 => Some(GeneratorFamily::Yager { p }),
            Self::Frank { s } if s > 0.0 => Some(GeneratorFamily::Frank { s }),
            Self::Dombi { p } if p > 0.0 => Some(GeneratorFamily::Dombi { p }),
            Self::SchweizerSklar { p } if p.is_finite() => {
                Some(GeneratorFamily::SchweizerSklar { p })
            }
            Self::SugenoWeber { p } if p > -1.0 && p.is_finite() => {
                Some(GeneratorFamily::SugenoWeber { p })
            }
            Self::AczelAlsina { p } if p > 0.0 && p.is_finite() => {
                Some(GeneratorFamily::AczelAlsina { p })
            }
            _ => None,
        }
    }

    /// Evaluate this family's t-conorm.
    #[inline]
    pub fn tconorm(self, a: f64, b: f64) -> f64 {
        tconorm(self, a, b)
    }

    /// Evaluate this family's t-conorm with a generic float type.
    #[inline]
    pub fn tconorm_generic<T: Float>(self, a: T, b: T) -> T {
        tconorm_generic(self, a, b)
    }

    /// Evaluate this family's standard-negation dual t-norm.
    ///
    /// Deprecated: a t-norm is not defined by its t-conorm. Use [`TNorm`] or
    /// [`LogicFamily::tnorm`]. The value is unchanged.
    #[deprecated(
        since = "0.2.2",
        note = "the t-norm is primary: use `TNorm` (for example `<Product as TNorm>::tnorm`) or `LogicFamily::tnorm`; the `TConormFamily` t-norm is scheduled for removal in 0.3.0"
    )]
    #[inline]
    pub fn tnorm(self, a: f64, b: f64) -> f64 {
        tnorm_dual(self, a, b)
    }

    /// Evaluate this family's standard-negation dual t-norm with a generic float type.
    ///
    /// Deprecated with [`TConormFamily::tnorm`].
    #[deprecated(
        since = "0.2.2",
        note = "the t-norm is primary: use `TNorm` or `LogicFamily::tnorm`; the `TConormFamily` t-norm is scheduled for removal in 0.3.0"
    )]
    #[inline]
    pub fn tnorm_generic<T: Float>(self, a: T, b: T) -> T {
        tnorm_dual_generic(self, a, b)
    }

    /// Evaluate the residuum `a -> b` when this family has a known path.
    ///
    /// Archimedean generator families use `f^-1(max(0, f(b) - f(a)))`.
    /// Godel, nilpotent minimum, and Mayor-Torrens use direct residuation.
    /// Drastic returns `None` because it is not left-continuous.
    pub fn residuum(self, a: f64, b: f64) -> Option<f64> {
        residuum(self, a, b)
    }

    /// Evaluate the t-norm gradient with respect to `(a, b)`.
    #[inline]
    pub fn tnorm_gradient(self, a: f64, b: f64) -> BinaryGradient<f64> {
        tnorm_gradient(self, a, b)
    }

    /// Evaluate the t-conorm gradient with respect to `(a, b)`.
    #[inline]
    pub fn tconorm_gradient(self, a: f64, b: f64) -> BinaryGradient<f64> {
        tconorm_gradient(self, a, b)
    }
}

/// Archimedean additive-generator family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeneratorFamily {
    /// Product generator `f(x) = -ln(x)`.
    Product,
    /// Lukasiewicz generator `f(x) = 1 - x`.
    Lukasiewicz,
    /// Hamacher generator with parameter `p >= 0`.
    Hamacher {
        /// Shape parameter.
        p: f64,
    },
    /// Frank generator with base `s > 0`.
    Frank {
        /// Base parameter. `s=1` uses the product limit.
        s: f64,
    },
    /// Yager generator with exponent `p > 0`.
    Yager {
        /// Shape parameter.
        p: f64,
    },
    /// Dombi generator with exponent `p > 0`.
    Dombi {
        /// Shape parameter.
        p: f64,
    },
    /// Schweizer-Sklar generator.
    SchweizerSklar {
        /// Shape parameter.
        p: f64,
    },
    /// Sugeno-Weber generator.
    SugenoWeber {
        /// Shape parameter, with valid finite values `p > -1`.
        p: f64,
    },
    /// Aczel-Alsina generator.
    AczelAlsina {
        /// Shape parameter.
        p: f64,
    },
}

impl GeneratorFamily {
    /// Evaluate the additive generator `f(x)`.
    pub fn value(self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Self::Product => {
                if x <= 0.0 {
                    f64::INFINITY
                } else {
                    -x.ln()
                }
            }
            Self::Lukasiewicz => 1.0 - x,
            Self::Hamacher { p } => hamacher_generator(x, p),
            Self::Frank { s } => frank_generator(x, s),
            Self::Yager { p } => (1.0 - x).powf(p),
            Self::Dombi { p } => {
                if x <= 0.0 {
                    f64::INFINITY
                } else if x >= 1.0 {
                    0.0
                } else {
                    ((1.0 - x) / x).powf(p)
                }
            }
            Self::SchweizerSklar { p } => {
                if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
                    Self::Product.value(x)
                } else if x <= 0.0 && p <= 0.0 {
                    f64::INFINITY
                } else {
                    (1.0 - x.powf(p)) / p
                }
            }
            Self::SugenoWeber { p } => {
                if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
                    1.0 - x
                } else {
                    1.0 - (p.mul_add(x, 1.0)).ln() / (1.0 + p).ln()
                }
            }
            Self::AczelAlsina { p } => {
                if x <= 0.0 {
                    f64::INFINITY
                } else {
                    (-x.ln()).powf(p)
                }
            }
        }
    }

    /// Evaluate the generator pseudo-inverse `f^-1(y)`.
    pub fn inverse(self, y: f64) -> f64 {
        let y = y.max(0.0);
        match self {
            Self::Product => (-y).exp(),
            Self::Lukasiewicz => (1.0 - y).clamp(0.0, 1.0),
            Self::Hamacher { p } => hamacher_generator_inverse(y, p),
            Self::Frank { s } => frank_generator_inverse(y, s),
            Self::Yager { p } => (1.0 - y.powf(1.0 / p)).clamp(0.0, 1.0),
            Self::Dombi { p } => 1.0 / (1.0 + y.powf(1.0 / p)),
            Self::SchweizerSklar { p } => {
                if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
                    Self::Product.inverse(y)
                } else {
                    (1.0 - p * y).max(0.0).powf(1.0 / p).clamp(0.0, 1.0)
                }
            }
            Self::SugenoWeber { p } => {
                if p.abs() <= FRANK_PRODUCT_LIMIT_EPS {
                    (1.0 - y).clamp(0.0, 1.0)
                } else {
                    (((1.0 + p).powf(1.0 - y) - 1.0) / p).clamp(0.0, 1.0)
                }
            }
            Self::AczelAlsina { p } => (-y.powf(1.0 / p)).exp(),
        }
    }

    /// Evaluate the t-norm induced by this generator.
    pub fn tnorm(self, a: f64, b: f64) -> f64 {
        self.inverse(self.value(a) + self.value(b))
    }

    /// Evaluate the generator residuum `a -> b`.
    ///
    /// This is `f^-1(f(b) - f(a))`. When `f(b)` is infinite, either `b` is the
    /// zero of a strict generator (`f(0) = inf`, and the result is
    /// `f^-1(inf) = 0`), or both values overflowed and the difference is
    /// taken in log space. Subtracting the overflowed values gave
    /// `inf - inf = NaN`, which `max(0, NaN)` turned into 0 and the inverse
    /// into 1.
    pub fn residuum(self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        if a <= b {
            return 1.0;
        }
        let difference = self.value(b) - self.value(a);
        if !difference.is_nan() && difference != f64::INFINITY {
            return self.inverse(difference.max(0.0));
        }
        if b <= 0.0 {
            return self.inverse(f64::INFINITY);
        }
        // ln(f(b) - f(a)) = ln f(b) + ln(1 - f(a) / f(b)), with f(a) < f(b).
        let (ln_fa, ln_fb) = (self.ln_value(a), self.ln_value(b));
        self.inverse_from_ln(ln_fb + ln_one_minus_exp((ln_fa - ln_fb).min(-f64::MIN_POSITIVE)))
    }

    /// `ln f(x)` for `0 < x < 1`, finite where `f(x)` overflows.
    fn ln_value(self, x: f64) -> f64 {
        match self {
            // f = ((1-x)/x)^p
            Self::Dombi { p } => p * ((-x).ln_1p() - x.ln()),
            // f = (-ln x)^p
            Self::AczelAlsina { p } => p * (-x.ln()).ln(),
            // f = (x^p - 1) / (-p) for p < 0, and x^p - 1 = e^t - 1 with
            // t = p ln x > 0; ln(e^t - 1) = t + ln(1 - e^-t).
            Self::SchweizerSklar { p } if p < -FRANK_PRODUCT_LIMIT_EPS => {
                let t = p * x.ln();
                t + ln_one_minus_exp(-t) - (-p).ln()
            }
            _ => self.value(x).ln(),
        }
    }

    /// `f^-1(e^l)`, finite where `e^l` overflows.
    fn inverse_from_ln(self, l: f64) -> f64 {
        match self {
            Self::Dombi { p } => 1.0 / (1.0 + (l / p).exp()),
            Self::AczelAlsina { p } => (-(l / p).exp()).exp(),
            // (1 - p y)^(1/p) with -p y = e^u, u = ln(-p) + l:
            // ln(1 + e^u) = max(u, 0) + ln_1p(e^-|u|).
            Self::SchweizerSklar { p } if p < -FRANK_PRODUCT_LIMIT_EPS => {
                let u = (-p).ln() + l;
                let softplus = u.max(0.0) + (-u.abs()).exp().ln_1p();
                (softplus / p).exp().clamp(0.0, 1.0)
            }
            _ => self.inverse(l.exp()),
        }
    }
}

/// Standard t-norm fuzzy logic families with residual implication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicFamily {
    /// Godel logic: minimum t-norm.
    Godel,
    /// Product logic: product t-norm.
    Product,
    /// Lukasiewicz logic: bounded-sum t-norm.
    Lukasiewicz,
}

/// Partial derivatives of a binary operation.
///
/// `da` is the derivative with respect to the first argument. `db` is the
/// derivative with respect to the second argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BinaryGradient<T> {
    /// Partial derivative with respect to the first argument.
    pub da: T,
    /// Partial derivative with respect to the second argument.
    pub db: T,
}

impl<T> BinaryGradient<T> {
    /// Create a binary gradient from the two partial derivatives.
    pub const fn new(da: T, db: T) -> Self {
        Self { da, db }
    }
}

impl LogicFamily {
    /// The t-conorm catalog family corresponding to this logic.
    pub const fn family(self) -> Family {
        match self {
            Self::Godel => GODEL,
            Self::Product => PRODUCT,
            Self::Lukasiewicz => LUKASIEWICZ,
        }
    }

    /// Evaluate the t-norm conjunction.
    ///
    /// This is the primary conjunction of the logic; [`crate::TNorm`] exposes
    /// the same function at the type level.
    pub fn tnorm(self, a: f64, b: f64) -> f64 {
        tnorm_dual(self.family(), a, b)
    }

    /// Evaluate the dual t-conorm disjunction.
    pub fn tconorm(self, a: f64, b: f64) -> f64 {
        tconorm(self.family(), a, b)
    }

    /// Evaluate the residual implication `a -> b`.
    pub fn residuum(self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => {
                if a <= b {
                    1.0
                } else {
                    b
                }
            }
            Self::Product => {
                if a <= b {
                    1.0
                } else {
                    b / a
                }
            }
            Self::Lukasiewicz => (1.0 - a + b).min(1.0),
        }
    }

    /// Evaluate the residual negation `a -> 0`.
    pub fn neg(self, a: f64) -> f64 {
        self.residuum(a, 0.0)
    }

    /// Evaluate the t-norm gradient with respect to `(a, b)`.
    ///
    /// The inputs are clamped to `[0, 1]` before selecting the local gradient.
    /// At non-differentiable points, min/max split ties evenly and
    /// Lukasiewicz hinges use the zero subgradient.
    pub fn tnorm_gradient(self, a: f64, b: f64) -> BinaryGradient<f64> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => min_gradient(a, b),
            Self::Product => BinaryGradient::new(b, a),
            Self::Lukasiewicz => {
                if a + b > 1.0 {
                    BinaryGradient::new(1.0, 1.0)
                } else {
                    BinaryGradient::new(0.0, 0.0)
                }
            }
        }
    }

    /// Evaluate the t-conorm gradient with respect to `(a, b)`.
    ///
    /// The inputs are clamped to `[0, 1]` before selecting the local gradient.
    /// At non-differentiable points, min/max split ties evenly and
    /// Lukasiewicz hinges use the zero subgradient.
    pub fn tconorm_gradient(self, a: f64, b: f64) -> BinaryGradient<f64> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => max_gradient(a, b),
            Self::Product => BinaryGradient::new(1.0 - b, 1.0 - a),
            Self::Lukasiewicz => {
                if a + b < 1.0 {
                    BinaryGradient::new(1.0, 1.0)
                } else {
                    BinaryGradient::new(0.0, 0.0)
                }
            }
        }
    }

    /// Evaluate the residual-implication gradient with respect to `(a, b)`.
    ///
    /// The inputs are clamped to `[0, 1]` before selecting the local gradient.
    /// On branch boundaries where the residuum saturates to `1`, this uses the
    /// saturated branch's zero subgradient.
    pub fn residuum_gradient(self, a: f64, b: f64) -> BinaryGradient<f64> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => {
                if a > b {
                    BinaryGradient::new(0.0, 1.0)
                } else {
                    BinaryGradient::new(0.0, 0.0)
                }
            }
            Self::Product => {
                if a > b {
                    BinaryGradient::new(-b / (a * a), 1.0 / a)
                } else {
                    BinaryGradient::new(0.0, 0.0)
                }
            }
            Self::Lukasiewicz => {
                if a > b {
                    BinaryGradient::new(-1.0, 1.0)
                } else {
                    BinaryGradient::new(0.0, 0.0)
                }
            }
        }
    }

    /// Evaluate the residual-negation gradient with respect to `a`.
    pub fn neg_gradient(self, _a: f64) -> f64 {
        match self {
            Self::Godel | Self::Product => 0.0,
            Self::Lukasiewicz => -1.0,
        }
    }

    /// Evaluate the t-norm conjunction using `f32`.
    pub fn tnorm_f32(self, a: f32, b: f32) -> f32 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => a.min(b),
            Self::Product => a * b,
            Self::Lukasiewicz => (a + b - 1.0).max(0.0),
        }
    }

    /// Evaluate the dual t-conorm disjunction using `f32`.
    pub fn tconorm_f32(self, a: f32, b: f32) -> f32 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => a.max(b),
            Self::Product => a + b - a * b,
            Self::Lukasiewicz => (a + b).min(1.0),
        }
    }

    /// Evaluate the residual implication using `f32`.
    pub fn residuum_f32(self, a: f32, b: f32) -> f32 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => {
                if a <= b {
                    1.0
                } else {
                    b
                }
            }
            Self::Product => {
                if a <= b {
                    1.0
                } else {
                    b / a
                }
            }
            Self::Lukasiewicz => (1.0 - a + b).min(1.0),
        }
    }

    /// Evaluate the residual negation using `f32`.
    pub fn neg_f32(self, a: f32) -> f32 {
        self.residuum_f32(a, 0.0)
    }

    /// Evaluate the t-norm gradient with respect to `(a, b)` using `f32`.
    pub fn tnorm_gradient_f32(self, a: f32, b: f32) -> BinaryGradient<f32> {
        let gradient = self.tnorm_gradient(f64::from(a), f64::from(b));
        BinaryGradient::new(gradient.da as f32, gradient.db as f32)
    }

    /// Evaluate the t-conorm gradient with respect to `(a, b)` using `f32`.
    pub fn tconorm_gradient_f32(self, a: f32, b: f32) -> BinaryGradient<f32> {
        let gradient = self.tconorm_gradient(f64::from(a), f64::from(b));
        BinaryGradient::new(gradient.da as f32, gradient.db as f32)
    }

    /// Evaluate the residual-implication gradient with respect to `(a, b)` using `f32`.
    pub fn residuum_gradient_f32(self, a: f32, b: f32) -> BinaryGradient<f32> {
        let gradient = self.residuum_gradient(f64::from(a), f64::from(b));
        BinaryGradient::new(gradient.da as f32, gradient.db as f32)
    }

    /// Evaluate the residual-negation gradient with respect to `a` using `f32`.
    pub fn neg_gradient_f32(self, a: f32) -> f32 {
        self.neg_gradient(f64::from(a)) as f32
    }

    /// Evaluate the t-norm conjunction for Burn tensors.
    ///
    /// This method is available with the `burn` feature. Inputs are clamped to
    /// `[0, 1]` before applying the operation.
    #[cfg(feature = "burn")]
    pub fn tnorm_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        self,
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => a.min_pair(b),
            Self::Product => a * b,
            Self::Lukasiewicz => (a + b - 1.0).clamp_min(0.0),
        }
    }

    /// Evaluate the dual t-conorm disjunction for Burn tensors.
    ///
    /// This method is available with the `burn` feature. Inputs are clamped to
    /// `[0, 1]` before applying the operation.
    #[cfg(feature = "burn")]
    pub fn tconorm_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        self,
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => a.max_pair(b),
            Self::Product => {
                let product = a.clone() * b.clone();
                a + b - product
            }
            Self::Lukasiewicz => (a + b).clamp_max(1.0),
        }
    }

    /// Evaluate the residual implication `a -> b` for Burn tensors.
    ///
    /// This method is available with the `burn` feature. Inputs are clamped to
    /// `[0, 1]` before applying the operation.
    #[cfg(feature = "burn")]
    pub fn residuum_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        self,
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Godel => {
                let mask = a.lower_equal(b.clone());
                let one = b.ones_like();
                b.mask_where(mask, one)
            }
            Self::Product => {
                let mask = a.clone().lower_equal(b.clone());
                let one = a.ones_like();
                let denominator = a.clamp_min(1e-12);
                let quotient = b / denominator;
                quotient.mask_where(mask, one)
            }
            Self::Lukasiewicz => (a.neg().add_scalar(1.0) + b).clamp_max(1.0),
        }
    }

    /// Evaluate the residual negation `a -> 0` for Burn tensors.
    ///
    /// This method is available with the `burn` feature. Inputs are clamped to
    /// `[0, 1]` before applying the operation.
    #[cfg(feature = "burn")]
    pub fn neg_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        self,
        a: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        let a = a.clamp(0.0, 1.0);
        match self {
            Self::Godel | Self::Product => {
                let zeros = a.zeros_like();
                let ones = a.ones_like();
                zeros.mask_where(a.lower_equal_elem(0.0), ones)
            }
            Self::Lukasiewicz => a.neg().add_scalar(1.0),
        }
    }
}

/// A left-continuous t-norm paired with its residuum.
///
/// In fuzzy logic the t-norm `⊗` and its residuum `→` are the primary adjoint
/// pair. The residuum is the largest `c` with `a ⊗ c <= b`, so
///
/// ```text
/// a ⊗ b <= c   iff   a <= (b → c)
/// ```
///
/// holds for every `a`, `b`, `c` in `[0, 1]`. The t-conorm is the dual of this
/// pair under a strong negation; [`TConormFamily`] is an aggregation catalog,
/// not the definition of the t-norm.
///
/// The implementors are the zero-sized logic markers [`Godel`], [`Product`]
/// and [`Lukasiewicz`]. They delegate to [`LogicFamily`], so every value is
/// bit-identical to the corresponding `LogicFamily` method.
///
/// ```
/// use tnorms::{LogicFamily, Product, TNorm};
///
/// assert_eq!(Product::NAME, "product");
/// assert_eq!(
///     <Product as TNorm>::tnorm(0.4, 0.7),
///     LogicFamily::Product.tnorm(0.4, 0.7)
/// );
/// assert!((<Product as TNorm>::residuum(0.7, 0.28) - 0.4).abs() < 1e-12);
/// assert!((<Product as TNorm>::tconorm(0.5, 0.6) - 0.8).abs() < 1e-12);
/// ```
///
/// [`TNorm::tnorm`], [`TNorm::residuum`] and [`TNorm::tconorm`] share their
/// names with [`TruthAlgebra`]. With both traits imported, an unqualified
/// `Product::tnorm(a, b)` is ambiguous; qualify it as
/// `<Product as TNorm>::tnorm(a, b)`, or import one trait at a time.
pub trait TNorm {
    /// Stable lower-case family name, for example `"product"`.
    const NAME: &'static str;

    /// Evaluate the t-norm conjunction `a ⊗ b`.
    fn tnorm(a: f64, b: f64) -> f64;

    /// Evaluate the residuum `a → b`, the right adjoint of the t-norm.
    fn residuum(a: f64, b: f64) -> f64;

    /// Evaluate the residual negation `¬a = a → 0`.
    ///
    /// This is the negation induced by the adjoint pair, not an arbitrary
    /// strong negation. It is involutive only for Łukasiewicz.
    fn negation(a: f64) -> f64 {
        Self::residuum(a, 0.0)
    }

    /// Evaluate the dual t-conorm `a ⊕ b = ¬(¬a ⊗ ¬b)`.
    ///
    /// The default is the duality identity for an implementor whose
    /// [`negation`](TNorm::negation) is strong (involutive). Godel and Product
    /// residuate to a non-involutive negation, under which this identity
    /// collapses to the constant `1`, so the built-in families override it and
    /// return the standard-negation dual instead.
    fn tconorm(a: f64, b: f64) -> f64 {
        Self::negation(Self::tnorm(Self::negation(a), Self::negation(b)))
    }
}

/// A standard residuated truth algebra on degrees in `[0, 1]`.
///
/// This trait is for type-level selection of the common fuzzy-logic families.
/// It names the t-norm conjunction, dual t-conorm disjunction, residual
/// implication, and residual negation. The methods are static because the
/// implementors are zero-sized marker types.
pub trait TruthAlgebra {
    /// The runtime logic family represented by this marker type.
    const FAMILY: LogicFamily;

    /// The fully true degree, `1.0`.
    #[inline]
    fn top() -> f64 {
        1.0
    }

    /// The fully false degree, `0.0`.
    #[inline]
    fn bottom() -> f64 {
        0.0
    }

    /// Evaluate the t-norm conjunction.
    #[inline]
    fn tnorm(a: f64, b: f64) -> f64 {
        Self::FAMILY.tnorm(a, b)
    }

    /// Evaluate the dual t-conorm disjunction.
    #[inline]
    fn tconorm(a: f64, b: f64) -> f64 {
        Self::FAMILY.tconorm(a, b)
    }

    /// Evaluate the residual implication `a -> b`.
    #[inline]
    fn residuum(a: f64, b: f64) -> f64 {
        Self::FAMILY.residuum(a, b)
    }

    /// Evaluate the residual negation `a -> 0`.
    #[inline]
    fn neg(a: f64) -> f64 {
        Self::FAMILY.neg(a)
    }

    /// Evaluate the t-norm gradient with respect to `(a, b)`.
    #[inline]
    fn tnorm_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
        Self::FAMILY.tnorm_gradient(a, b)
    }

    /// Evaluate the dual t-conorm gradient with respect to `(a, b)`.
    #[inline]
    fn tconorm_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
        Self::FAMILY.tconorm_gradient(a, b)
    }

    /// Evaluate the residual-implication gradient with respect to `(a, b)`.
    #[inline]
    fn residuum_gradient(a: f64, b: f64) -> BinaryGradient<f64> {
        Self::FAMILY.residuum_gradient(a, b)
    }

    /// Evaluate the residual-negation gradient with respect to `a`.
    #[inline]
    fn neg_gradient(a: f64) -> f64 {
        Self::FAMILY.neg_gradient(a)
    }

    /// The fully true `f32` degree, `1.0`.
    #[inline]
    fn top_f32() -> f32 {
        1.0
    }

    /// The fully false `f32` degree, `0.0`.
    #[inline]
    fn bottom_f32() -> f32 {
        0.0
    }

    /// Evaluate the t-norm conjunction using `f32`.
    #[inline]
    fn tnorm_f32(a: f32, b: f32) -> f32 {
        Self::FAMILY.tnorm_f32(a, b)
    }

    /// Evaluate the dual t-conorm disjunction using `f32`.
    #[inline]
    fn tconorm_f32(a: f32, b: f32) -> f32 {
        Self::FAMILY.tconorm_f32(a, b)
    }

    /// Evaluate the residual implication using `f32`.
    #[inline]
    fn residuum_f32(a: f32, b: f32) -> f32 {
        Self::FAMILY.residuum_f32(a, b)
    }

    /// Evaluate the residual negation using `f32`.
    #[inline]
    fn neg_f32(a: f32) -> f32 {
        Self::FAMILY.neg_f32(a)
    }

    /// Evaluate the t-norm gradient with respect to `(a, b)` using `f32`.
    #[inline]
    fn tnorm_gradient_f32(a: f32, b: f32) -> BinaryGradient<f32> {
        Self::FAMILY.tnorm_gradient_f32(a, b)
    }

    /// Evaluate the dual t-conorm gradient with respect to `(a, b)` using `f32`.
    #[inline]
    fn tconorm_gradient_f32(a: f32, b: f32) -> BinaryGradient<f32> {
        Self::FAMILY.tconorm_gradient_f32(a, b)
    }

    /// Evaluate the residual-implication gradient with respect to `(a, b)` using `f32`.
    #[inline]
    fn residuum_gradient_f32(a: f32, b: f32) -> BinaryGradient<f32> {
        Self::FAMILY.residuum_gradient_f32(a, b)
    }

    /// Evaluate the residual-negation gradient with respect to `a` using `f32`.
    #[inline]
    fn neg_gradient_f32(a: f32) -> f32 {
        Self::FAMILY.neg_gradient_f32(a)
    }

    /// Evaluate the t-norm conjunction for Burn tensors.
    #[cfg(feature = "burn")]
    #[inline]
    fn tnorm_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        Self::FAMILY.tnorm_tensor(a, b)
    }

    /// Evaluate the dual t-conorm disjunction for Burn tensors.
    #[cfg(feature = "burn")]
    #[inline]
    fn tconorm_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        Self::FAMILY.tconorm_tensor(a, b)
    }

    /// Evaluate the residual implication `a -> b` for Burn tensors.
    #[cfg(feature = "burn")]
    #[inline]
    fn residuum_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        a: burn::tensor::Tensor<B, D>,
        b: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        Self::FAMILY.residuum_tensor(a, b)
    }

    /// Evaluate the residual negation `a -> 0` for Burn tensors.
    #[cfg(feature = "burn")]
    #[inline]
    fn neg_tensor<B: burn::tensor::backend::Backend, const D: usize>(
        a: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        Self::FAMILY.neg_tensor(a)
    }
}

/// Godel truth algebra: minimum t-norm, maximum t-conorm, and Godel residuum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Godel;

impl TruthAlgebra for Godel {
    const FAMILY: LogicFamily = LogicFamily::Godel;
}

/// Product truth algebra: product t-norm, probabilistic t-conorm, and Goguen residuum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Product;

impl TruthAlgebra for Product {
    const FAMILY: LogicFamily = LogicFamily::Product;
}

/// Lukasiewicz truth algebra: bounded-sum t-norm, bounded t-conorm, and Lukasiewicz residuum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lukasiewicz;

impl TruthAlgebra for Lukasiewicz {
    const FAMILY: LogicFamily = LogicFamily::Lukasiewicz;
}

impl TNorm for Godel {
    const NAME: &'static str = "godel";

    #[inline]
    fn tnorm(a: f64, b: f64) -> f64 {
        LogicFamily::Godel.tnorm(a, b)
    }

    #[inline]
    fn residuum(a: f64, b: f64) -> f64 {
        LogicFamily::Godel.residuum(a, b)
    }

    #[inline]
    fn negation(a: f64) -> f64 {
        LogicFamily::Godel.neg(a)
    }

    #[inline]
    fn tconorm(a: f64, b: f64) -> f64 {
        LogicFamily::Godel.tconorm(a, b)
    }
}

impl TNorm for Product {
    const NAME: &'static str = "product";

    #[inline]
    fn tnorm(a: f64, b: f64) -> f64 {
        LogicFamily::Product.tnorm(a, b)
    }

    #[inline]
    fn residuum(a: f64, b: f64) -> f64 {
        LogicFamily::Product.residuum(a, b)
    }

    #[inline]
    fn negation(a: f64) -> f64 {
        LogicFamily::Product.neg(a)
    }

    #[inline]
    fn tconorm(a: f64, b: f64) -> f64 {
        LogicFamily::Product.tconorm(a, b)
    }
}

impl TNorm for Lukasiewicz {
    const NAME: &'static str = "lukasiewicz";

    #[inline]
    fn tnorm(a: f64, b: f64) -> f64 {
        LogicFamily::Lukasiewicz.tnorm(a, b)
    }

    #[inline]
    fn residuum(a: f64, b: f64) -> f64 {
        LogicFamily::Lukasiewicz.residuum(a, b)
    }

    #[inline]
    fn negation(a: f64) -> f64 {
        LogicFamily::Lukasiewicz.neg(a)
    }

    #[inline]
    fn tconorm(a: f64, b: f64) -> f64 {
        LogicFamily::Lukasiewicz.tconorm(a, b)
    }
}

/// The dual t-norm T(a,b) = 1 - S(1-a, 1-b) for a given t-conorm family.
///
/// T-norms generalize logical AND. The t-norm is primary, so this
/// t-conorm-first route is deprecated in favour of [`TNorm`] or
/// [`LogicFamily::tnorm`]. The value is unchanged.
#[deprecated(
    since = "0.2.2",
    note = "the t-norm is primary: use `TNorm` (for example `<Product as TNorm>::tnorm`) or `LogicFamily::tnorm`; the `TConormFamily` t-norm is scheduled for removal in 0.3.0"
)]
pub fn tnorm(family: TConormFamily, a: f64, b: f64) -> f64 {
    tnorm_dual(family, a, b)
}

/// Evaluate the standard-negation dual t-norm for a generic float type.
///
/// Deprecated with [`tnorm`]: the t-norm is primary and should not be derived
/// from a t-conorm family.
#[deprecated(
    since = "0.2.2",
    note = "the t-norm is primary: use `TNorm` or `LogicFamily::tnorm`; the `TConormFamily` t-norm is scheduled for removal in 0.3.0"
)]
pub fn tnorm_generic<T: Float>(family: TConormFamily, a: T, b: T) -> T {
    tnorm_dual_generic(family, a, b)
}

/// Non-deprecated implementation behind [`tnorm`].
pub(crate) fn tnorm_dual(family: TConormFamily, a: f64, b: f64) -> f64 {
    tnorm_dual_generic(family, a, b)
}

fn tnorm_dual_generic<T: Float>(family: TConormFamily, a: T, b: T) -> T {
    let a = clamp01(a);
    let b = clamp01(b);
    match family {
        TConormFamily::Maximum => a.min(b),
        TConormFamily::Probabilistic => a * b,
        TConormFamily::Bounded => (a + b - T::one()).max(T::zero()),
        TConormFamily::Drastic => drastic_tnorm(a, b),
        TConormFamily::Hamacher => hamacher_tnorm(a, b),
        TConormFamily::NilpotentMinimum => nilpotent_minimum_tnorm(a, b),
        TConormFamily::SchweizerSklar { p } => schweizer_sklar_tnorm(a, b, cast(p)),
        TConormFamily::SugenoWeber { p } => sugeno_weber_tnorm(a, b, cast(p)),
        TConormFamily::AczelAlsina { p } => aczel_alsina_tnorm(a, b, cast(p)),
        TConormFamily::MayorTorrens { p } => mayor_torrens_tnorm(a, b, cast(p)),
        _ => clamp_result(T::one() - tconorm_generic(family, T::one() - a, T::one() - b)),
    }
}

/// Evaluate the residuum `a -> b` for a t-norm family when available.
///
/// This returns `None` for the drastic product because it is not
/// left-continuous. Exact parameter limits are dispatched to the corresponding
/// named logic rather than evaluated through a degenerate generator.
pub fn residuum(family: TConormFamily, a: f64, b: f64) -> Option<f64> {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    if a <= b {
        return Some(1.0);
    }

    match family {
        TConormFamily::Maximum => Some(b),
        TConormFamily::NilpotentMinimum => Some((1.0 - a).max(b)),
        TConormFamily::Drastic => None,
        TConormFamily::MayorTorrens { p } => {
            let p = p.clamp(0.0, 1.0);
            Some(if a <= p { p - a + b } else { b })
        }
        TConormFamily::SchweizerSklar { p } if p == f64::NEG_INFINITY => Some(b),
        TConormFamily::SugenoWeber { p } if p == f64::INFINITY => Some(b / a),
        TConormFamily::AczelAlsina { p } | TConormFamily::Yager { p } if p == f64::INFINITY => {
            Some(b)
        }
        TConormFamily::Frank { s } if s > 0.0 && s.ln() <= FRANK_MAXIMUM_LIMIT_LOG => Some(b),
        TConormFamily::Frank { s } if s > 0.0 && s.ln() >= FRANK_BOUNDED_LIMIT_LOG => {
            Some(1.0 - a + b)
        }
        _ => family.generator().map(|generator| generator.residuum(a, b)),
    }
}

/// Numerically approximate the residuum as `sup { c : T(a, c) <= b }`.
pub fn residuum_numeric(family: TConormFamily, a: f64, b: f64, iterations: usize) -> f64 {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    if a <= b {
        return 1.0;
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..iterations {
        let mid = (lo + hi) * 0.5;
        if tnorm_dual(family, a, mid) <= b {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Evaluate the t-norm gradient with respect to `(a, b)`.
pub fn tnorm_gradient(family: TConormFamily, a: f64, b: f64) -> BinaryGradient<f64> {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    match family {
        TConormFamily::Maximum => min_gradient(a, b),
        TConormFamily::Probabilistic => BinaryGradient::new(b, a),
        TConormFamily::Bounded => {
            if a + b > 1.0 {
                BinaryGradient::new(1.0, 1.0)
            } else {
                BinaryGradient::new(0.0, 0.0)
            }
        }
        TConormFamily::Einstein => hamacher_tnorm_gradient(a, b, 2.0),
        TConormFamily::Hamacher => hamacher_tnorm_gradient(a, b, 0.0),
        TConormFamily::Yager { p } => yager_tnorm_gradient(a, b, p),
        TConormFamily::Frank { s } => frank_tnorm_gradient(a, b, s),
        TConormFamily::Dombi { p } => dombi_tnorm_gradient(a, b, p),
        TConormFamily::Drastic => drastic_tnorm_gradient(a, b),
        TConormFamily::NilpotentMinimum => nilpotent_minimum_tnorm_gradient(a, b),
        TConormFamily::SchweizerSklar { p } => schweizer_sklar_tnorm_gradient(a, b, p),
        TConormFamily::SugenoWeber { p } => sugeno_weber_tnorm_gradient(a, b, p),
        TConormFamily::AczelAlsina { p } => aczel_alsina_tnorm_gradient(a, b, p),
        TConormFamily::MayorTorrens { p } => mayor_torrens_tnorm_gradient(a, b, p),
    }
}

/// Evaluate the t-conorm gradient with respect to `(a, b)`.
pub fn tconorm_gradient(family: TConormFamily, a: f64, b: f64) -> BinaryGradient<f64> {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    let dual = tnorm_gradient(family, 1.0 - a, 1.0 - b);
    BinaryGradient::new(dual.da, dual.db)
}

/// Aggregate multiple values using a t-conorm (generalized multi-way OR).
///
/// Folds left-to-right using associativity: S(S(S(a, b), c), d)...
pub fn tconorm_fold(family: TConormFamily, values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0, // identity element for t-conorms
        1 => values[0].clamp(0.0, 1.0),
        _ => values
            .iter()
            .skip(1)
            .fold(values[0].clamp(0.0, 1.0), |acc, &v| tconorm(family, acc, v)),
    }
}

/// Aggregate multiple values using a t-norm (generalized multi-way AND).
pub fn tnorm_fold(family: TConormFamily, values: &[f64]) -> f64 {
    match values.len() {
        0 => 1.0, // identity element for t-norms
        1 => values[0].clamp(0.0, 1.0),
        _ => values
            .iter()
            .skip(1)
            .fold(values[0].clamp(0.0, 1.0), |acc, &v| {
                tnorm_dual(family, acc, v)
            }),
    }
}

#[cfg(test)]
mod tests;
