//! Behaviour outside `[0, 1]` and on NaN, per family.
//!
//! Every family clamps its inputs to `[0, 1]`, so an out-of-range input must
//! give exactly the value at the nearest endpoint. NaN must never panic, and
//! the t-norm and t-conorm of any family must stay inside `[0, 1]`.

#![allow(deprecated)] // exercises the deprecated catalog t-norm

use tnorms::{
    power_mean_weighted, residuum, tconorm, tconorm_fold, tconorm_gradient, tnorm, tnorm_fold,
    tnorm_gradient, BinaryGradient, LogicFamily, TConormFamily,
};

fn all_families() -> Vec<TConormFamily> {
    vec![
        TConormFamily::Maximum,
        TConormFamily::Probabilistic,
        TConormFamily::Bounded,
        TConormFamily::Einstein,
        TConormFamily::Hamacher,
        TConormFamily::Yager { p: 2.0 },
        TConormFamily::Frank { s: 2.0 },
        TConormFamily::Frank { s: 1e-12 },
        TConormFamily::Dombi { p: 2.0 },
        TConormFamily::Drastic,
        TConormFamily::NilpotentMinimum,
        TConormFamily::SchweizerSklar { p: -1.0 },
        TConormFamily::SchweizerSklar { p: 0.5 },
        TConormFamily::SugenoWeber { p: 0.5 },
        TConormFamily::AczelAlsina { p: 2.0 },
        TConormFamily::MayorTorrens { p: 0.4 },
    ]
}

const PAIRS: [(f64, f64, f64, f64); 6] = [
    // (a, b, clamped a, clamped b)
    (1.5, 0.3, 1.0, 0.3),
    (-0.2, 0.6, 0.0, 0.6),
    (0.4, 7.0, 0.4, 1.0),
    (0.7, -3.0, 0.7, 0.0),
    (f64::INFINITY, 0.2, 1.0, 0.2),
    (0.2, f64::NEG_INFINITY, 0.2, 0.0),
];

fn same(x: f64, y: f64) -> bool {
    x == y || (x.is_nan() && y.is_nan())
}

fn same_grad(x: BinaryGradient<f64>, y: BinaryGradient<f64>) -> bool {
    same(x.da, y.da) && same(x.db, y.db)
}

#[test]
fn out_of_range_inputs_are_clamped_for_every_family() {
    for f in all_families() {
        for (a, b, ca, cb) in PAIRS {
            assert!(
                same(tnorm(f, a, b), tnorm(f, ca, cb)),
                "{f:?} tnorm({a}, {b})"
            );
            assert!(
                same(tconorm(f, a, b), tconorm(f, ca, cb)),
                "{f:?} tconorm({a}, {b})"
            );
            match (residuum(f, a, b), residuum(f, ca, cb)) {
                (Some(x), Some(y)) => assert!(same(x, y), "{f:?} residuum({a}, {b})"),
                (None, None) => {}
                other => panic!("{f:?} residuum({a}, {b}): {other:?}"),
            }
            assert!(
                same_grad(tnorm_gradient(f, a, b), tnorm_gradient(f, ca, cb)),
                "{f:?} tnorm_gradient({a}, {b})"
            );
            assert!(
                same_grad(tconorm_gradient(f, a, b), tconorm_gradient(f, ca, cb)),
                "{f:?} tconorm_gradient({a}, {b})"
            );
            assert!(same(tnorm_fold(f, &[a, b]), tnorm_fold(f, &[ca, cb])));
            assert!(same(tconorm_fold(f, &[a, b]), tconorm_fold(f, &[ca, cb])));
        }
    }
    for lf in [
        LogicFamily::Godel,
        LogicFamily::Product,
        LogicFamily::Lukasiewicz,
    ] {
        for (a, b, ca, cb) in PAIRS {
            assert!(same(lf.tnorm(a, b), lf.tnorm(ca, cb)), "{lf:?}");
            assert!(same(lf.tconorm(a, b), lf.tconorm(ca, cb)), "{lf:?}");
            assert!(same(lf.residuum(a, b), lf.residuum(ca, cb)), "{lf:?}");
        }
    }
}

#[test]
fn nan_inputs_do_not_panic_and_tnorm_tconorm_stay_in_unit_interval() {
    let nan = f64::NAN;
    for f in all_families() {
        for (a, b) in [(nan, 0.4), (0.4, nan), (nan, nan)] {
            let t = tnorm(f, a, b);
            let s = tconorm(f, a, b);
            assert!((0.0..=1.0).contains(&t), "{f:?} tnorm({a}, {b}) = {t}");
            assert!((0.0..=1.0).contains(&s), "{f:?} tconorm({a}, {b}) = {s}");
            let _ = residuum(f, a, b);
            let _ = tnorm_gradient(f, a, b);
            let _ = tconorm_gradient(f, a, b);
            let _ = tnorm_fold(f, &[a, b, 0.5]);
            let _ = tconorm_fold(f, &[a, b, 0.5]);
        }
    }
}

/// Weighted power mean M_p = (sum w_i x_i^p)^(1/p) with weights (0.25, 0.75)
/// on (0.25, 1.0): p = 2 gives sqrt(0.765625) = 0.875, p = -1 gives
/// 1 / (0.25/0.25 + 0.75/1) = 4/7, and p -> 0 gives 0.25^0.25 = 2^-0.5.
#[test]
fn weighted_power_mean_matches_definition_for_p_other_than_one() {
    let w = |i: usize| if i == 0 { 0.25 } else { 0.75 };
    let x = [0.25, 1.0];
    assert!((power_mean_weighted(&x, 2.0, w).unwrap() - 0.875).abs() < 1e-12);
    assert!((power_mean_weighted(&x, -1.0, w).unwrap() - 4.0 / 7.0).abs() < 1e-12);
    assert!((power_mean_weighted(&x, 0.0, w).unwrap() - 0.5f64.sqrt()).abs() < 1e-12);
}

/// T-norms and t-conorms map `[0, 1]^2` into `[0, 1]`, and the absorbing
/// elements hold to rounding: T(0, b) = 0 and S(1, b) = 1.
#[test]
fn outputs_stay_in_unit_interval_and_hit_boundaries_exactly() {
    let grid: Vec<f64> = (0..=20).map(|k| k as f64 / 20.0).collect();
    for f in all_families() {
        for &a in &grid {
            for &b in &grid {
                let t = tnorm(f, a, b);
                let s = tconorm(f, a, b);
                assert!((0.0..=1.0).contains(&t), "{f:?} tnorm({a}, {b}) = {t}");
                assert!((0.0..=1.0).contains(&s), "{f:?} tconorm({a}, {b}) = {s}");
            }
        }
        for &b in &grid {
            let t0 = tnorm(f, 0.0, b);
            let s1 = tconorm(f, 1.0, b);
            assert!(t0.abs() <= 1e-12, "{f:?} T(0, {b}) = {t0}");
            assert!((s1 - 1.0).abs() <= 1e-12, "{f:?} S(1, {b}) = {s1}");
        }
    }
}
