//! Boundary identities and accuracy where a formula can cancel or overflow.
//!
//! Every t-norm satisfies `T(x, 1) = x` and `T(x, 0) = 0`, and every t-conorm
//! `S(x, 0) = x` and `S(x, 1) = 1` (Klement, Mesiar & Pap, *Triangular
//! Norms*, 2000, Definition 1.1). These hold over a grid that includes values
//! within `1e-12` of 0 and 1, where the naive closed forms lose digits.

use tnorms::{residuum, tconorm, tnorm, GeneratorFamily, TConormFamily};

/// 0, 1/1000, ..., 1, plus points close to 0 and 1.
fn grid() -> Vec<f64> {
    let mut xs: Vec<f64> = (0..=1000).map(|i| f64::from(i) / 1000.0).collect();
    xs.extend([
        1e-300,
        1e-200,
        1e-20,
        1e-12,
        1e-6,
        0.05,
        1.0 - 1e-6,
        1.0 - 1e-12,
    ]);
    xs
}

fn assert_close(actual: f64, expected: f64, rel: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{what}: got {actual}, expected {expected}"
    );
}

#[test]
fn hamacher_identities_are_exact() {
    let h = TConormFamily::Hamacher;
    for x in grid() {
        assert_eq!(tnorm(h, x, 0.0), 0.0, "T({x}, 0)");
        assert_eq!(tnorm(h, 0.0, x), 0.0, "T(0, {x})");
        assert_eq!(tnorm(h, x, 1.0), x, "T({x}, 1)");
        assert_eq!(tnorm(h, 1.0, x), x, "T(1, {x})");
        assert_eq!(tconorm(h, x, 1.0), 1.0, "S({x}, 1)");
        assert_eq!(tconorm(h, 1.0, x), 1.0, "S(1, {x})");
        assert_eq!(tconorm(h, x, 0.0), x, "S({x}, 0)");
        assert_eq!(tconorm(h, 0.0, x), x, "S(0, {x})");
    }
}

#[test]
fn hamacher_is_accurate_near_the_corners() {
    let h = TConormFamily::Hamacher;
    for e in [1e-4, 1e-8, 1e-12] {
        // Hamacher product T(x, y) = xy / (x + y - xy); at x = y = e this is
        // e / (2 - e), and S(1-e, 1-e) = 1 - T(e, e).
        let t = e / (2.0 - e);
        assert_close(tnorm(h, e, e), t, 1e-14, "T(e, e)");
        let s = tconorm(h, 1.0 - e, 1.0 - e);
        assert!(
            (s - (1.0 - t)).abs() <= 2.0 * f64::EPSILON,
            "S(1-e, 1-e) = {s}, expected {}",
            1.0 - t
        );
    }
    // Interior agreement with the textbook rational form.
    for &(a, b) in &[(0.3, 0.7), (0.5, 0.5), (0.9, 0.2), (0.01, 0.99)] {
        assert_close(tnorm(h, a, b), a * b / (a + b - a * b), 1e-15, "T interior");
        assert_close(
            tconorm(h, a, b),
            (a + b - 2.0 * a * b) / (1.0 - a * b),
            1e-15,
            "S interior",
        );
    }
}

#[test]
fn schweizer_sklar_identities_hold_for_large_and_negative_p() {
    for p in [0.5, 2.0, 10.0, 20.0, 50.0, -0.5, -2.0, -10.0] {
        let f = TConormFamily::SchweizerSklar { p };
        for x in grid() {
            assert_eq!(tnorm(f, x, 0.0), 0.0, "p={p}: T({x}, 0)");
            assert_eq!(tnorm(f, 0.0, x), 0.0, "p={p}: T(0, {x})");
            assert_eq!(tnorm(f, x, 1.0), x, "p={p}: T({x}, 1)");
            assert_eq!(tnorm(f, 1.0, x), x, "p={p}: T(1, {x})");
        }
    }
}

#[test]
fn schweizer_sklar_matches_closed_form_in_the_interior() {
    // T(x, y) = (x^p + y^p - 1)^(1/p), evaluated where the direct form is
    // well conditioned.
    for p in [0.5, 2.0, 10.0, -0.5, -2.0, -10.0] {
        let f = TConormFamily::SchweizerSklar { p };
        for &(a, b) in &[(0.9_f64, 0.95_f64), (0.7, 0.8), (0.3, 0.4), (0.99, 0.5)] {
            let inner: f64 = a.powf(p) + b.powf(p) - 1.0;
            let expected = if inner <= 0.0 {
                0.0
            } else {
                inner.powf(1.0 / p)
            };
            let got = tnorm(f, a, b);
            assert!(
                (got - expected).abs() <= 1e-13,
                "p={p}: T({a}, {b}) = {got}, expected {expected}"
            );
        }
    }
}

const STRICT_GENERATORS: &[GeneratorFamily] = &[
    GeneratorFamily::Product,
    GeneratorFamily::Hamacher { p: 0.0 },
    GeneratorFamily::Hamacher { p: 2.0 },
    GeneratorFamily::Frank { s: 2.0 },
    GeneratorFamily::Frank { s: 1e-20 },
    GeneratorFamily::Dombi { p: 1.0 },
    GeneratorFamily::Dombi { p: 50.0 },
    GeneratorFamily::AczelAlsina { p: 2.0 },
    GeneratorFamily::AczelAlsina { p: 50.0 },
    GeneratorFamily::SchweizerSklar { p: -1.0 },
    GeneratorFamily::SchweizerSklar { p: -20.0 },
];

#[test]
fn strict_generator_residuum_to_zero_is_zero() {
    // For a strict t-norm T(a, c) > 0 whenever a, c > 0, so a -> 0 is 0 for
    // every a > 0, even when f(a) overflows like f(0) = inf.
    for &g in STRICT_GENERATORS {
        for a in grid().into_iter().filter(|&a| a > 0.0) {
            assert_eq!(g.residuum(a, 0.0), 0.0, "{g:?}: residuum({a}, 0)");
        }
    }
    assert_eq!(
        residuum(TConormFamily::Dombi { p: 50.0 }, 1e-12, 0.0),
        Some(0.0)
    );
}

#[test]
fn dombi_residuum_with_overflowing_generator_values() {
    // f(x) = ((1-x)/x)^p overflows for both arguments. The residuum c solves
    // f(c) = f(b) - f(a), so with g(x) = (1-x)/x,
    // g(c) = g(b) (1 - (g(a)/g(b))^p)^(1/p).
    let p = 50.0;
    let g = |x: f64| (1.0 - x) / x;
    for &(a, b) in &[(2e-12, 1e-12), (1e-6, 5e-7), (1e-9, 1e-12)] {
        let ratio: f64 = (g(a) / g(b)).powf(p);
        let gc = g(b) * (1.0 - ratio).powf(1.0 / p);
        let expected = 1.0 / (1.0 + gc);
        let got = GeneratorFamily::Dombi { p }.residuum(a, b);
        assert_close(got, expected, 1e-9, "Dombi residuum");
    }
}

#[test]
fn aczel_alsina_residuum_with_overflowing_generator_values() {
    // f(x) = (-ln x)^p; with h(x) = -ln x, h(c) = h(b) (1 - (h(a)/h(b))^p)^(1/p).
    let p = 200.0;
    let h = |x: f64| -x.ln();
    for &(a, b) in &[(1e-300, 1e-305), (1e-250, 1e-300)] {
        assert!(GeneratorFamily::AczelAlsina { p }.value(a).is_infinite());
        let ratio: f64 = (h(a) / h(b)).powf(p);
        let expected = (-h(b) * (1.0 - ratio).powf(1.0 / p)).exp();
        let got = GeneratorFamily::AczelAlsina { p }.residuum(a, b);
        assert_close(got, expected, 1e-9, "Aczel-Alsina residuum");
    }
}

#[test]
fn schweizer_sklar_residuum_with_overflowing_generator_values() {
    // f(x) = (x^p - 1) / (-p), p < 0, so c^p = b^p - a^p + 1 and
    // c = b (1 - (a/b)^p + b^-p)^(1/p).
    let p = -20.0;
    for &(a, b) in &[(1e-16, 1e-17), (1e-17, 5e-18)] {
        assert!(GeneratorFamily::SchweizerSklar { p }.value(a).is_infinite());
        let inner: f64 = 1.0 - (a / b).powf(p) + b.powf(-p);
        let expected = b * inner.powf(1.0 / p);
        let got = GeneratorFamily::SchweizerSklar { p }.residuum(a, b);
        assert_close(got, expected, 1e-9, "Schweizer-Sklar residuum");
    }
}

/// Frank t-norm for `s < 1` in a form with no cancellation:
/// `1 + (s^a-1)(s^b-1)/(s-1) = ((s^a - s) + s^b (1 - s^a)) / (1 - s)`.
fn frank_reference(a: f64, b: f64, s: f64) -> f64 {
    let ln_s = s.ln();
    let sa = (a * ln_s).exp();
    let sb = (b * ln_s).exp();
    let inner = ((sa - s) + sb * (1.0 - sa)) / (1.0 - s);
    inner.ln() / ln_s
}

#[test]
fn frank_generator_tnorm_is_accurate_for_small_s() {
    for s in [1e-200, 1e-20, 1e-5] {
        let g = GeneratorFamily::Frank { s };
        for &(a, b) in &[(0.1, 0.1), (0.6, 0.6), (0.3, 0.9), (0.5, 0.05)] {
            let got = g.tnorm(a, b);
            assert!((0.0..=1.0).contains(&got), "s={s}: T({a}, {b}) = {got}");
            assert_close(got, frank_reference(a, b, s), 1e-9, "Frank generator T");
        }
    }
    // ln(2e-12 - 1e-24) / ln(1e-20)
    let t = GeneratorFamily::Frank { s: 1e-20 }.tnorm(0.6, 0.6);
    assert!((t - 0.584_948_500_3).abs() < 1e-9, "T(0.6, 0.6) = {t}");
}

#[test]
fn frank_generator_identities() {
    for s in [1e-200, 1e-20, 1e-5, 0.5, 2.0, 1e5] {
        let g = GeneratorFamily::Frank { s };
        for x in grid() {
            assert_eq!(g.tnorm(x, 0.0), 0.0, "s={s}: T({x}, 0)");
            let t = g.tnorm(x, 1.0);
            assert!(
                (t - x).abs() <= 1e-9 * x.max(1e-300),
                "s={s}: T({x}, 1) = {t}"
            );
        }
    }
}
