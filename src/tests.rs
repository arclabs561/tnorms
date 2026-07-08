use super::*;
use proptest::prelude::*;

fn all_families() -> Vec<TConormFamily> {
    vec![
        TConormFamily::Maximum,
        TConormFamily::Probabilistic,
        TConormFamily::Bounded,
        TConormFamily::Einstein,
        TConormFamily::Hamacher,
        TConormFamily::Yager { p: 2.0 },
        TConormFamily::Frank { s: 2.0 },
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

#[test]
fn identity_element() {
    for f in all_families() {
        let r = tconorm(f, 0.5, 0.0);
        assert!(
            (r - 0.5).abs() < 1e-6,
            "{f:?}: S(0.5, 0) = {r}, expected 0.5"
        );
    }
}

#[test]
fn commutativity() {
    for f in all_families() {
        let r1 = tconorm(f, 0.3, 0.7);
        let r2 = tconorm(f, 0.7, 0.3);
        assert!(
            (r1 - r2).abs() < 1e-10,
            "{f:?}: S(0.3,0.7)={r1} != S(0.7,0.3)={r2}"
        );
    }
}

#[test]
fn monotonicity() {
    for f in all_families() {
        let r1 = tconorm(f, 0.3, 0.5);
        let r2 = tconorm(f, 0.6, 0.5);
        assert!(
            r2 >= r1 - 1e-10,
            "{f:?}: monotonicity violated S(0.3,0.5)={r1} > S(0.6,0.5)={r2}"
        );
    }
}

#[test]
fn bounded_output() {
    for f in all_families() {
        for &a in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            for &b in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                let r = tconorm(f, a, b);
                assert!(
                    (-1e-10..=1.0 + 1e-10).contains(&r),
                    "{f:?}: S({a},{b}) = {r} out of [0,1]"
                );
            }
        }
    }
}

#[test]
fn probabilistic_known_values() {
    let f = TConormFamily::Probabilistic;
    // S(0.5, 0.5) = 0.5 + 0.5 - 0.25 = 0.75
    assert!((tconorm(f, 0.5, 0.5) - 0.75).abs() < 1e-10);
    // S(1, x) = 1 for all x
    assert!((tconorm(f, 1.0, 0.3) - 1.0).abs() < 1e-10);
}

#[test]
fn yager_p1_equals_bounded() {
    let yager = TConormFamily::Yager { p: 1.0 };
    let bounded = TConormFamily::Bounded;
    for &a in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        for &b in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let ry = tconorm(yager, a, b);
            let rb = tconorm(bounded, a, b);
            assert!(
                (ry - rb).abs() < 1e-10,
                "Yager(p=1) != Bounded: S({a},{b}) = {ry} vs {rb}"
            );
        }
    }
}

#[test]
fn frank_s1_equals_probabilistic_limit() {
    let probabilistic = TConormFamily::Probabilistic;
    for &s in &[1.0, 1.0 - 1e-9, 1.0 + 1e-9] {
        let frank = TConormFamily::Frank { s };
        for &a in &[0.1, 0.4, 0.8] {
            for &b in &[0.2, 0.5, 0.9] {
                let actual = tconorm(frank, a, b);
                let expected = tconorm(probabilistic, a, b);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "Frank(s={s}) limit failed for S({a},{b}): {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn frank_extreme_s_uses_closed_limits() {
    let maximum = TConormFamily::Maximum;
    let bounded = TConormFamily::Bounded;
    let small = TConormFamily::Frank {
        s: f64::MIN_POSITIVE,
    };
    let large = TConormFamily::Frank { s: f64::MAX };

    for &a in &[0.1, 0.4, 0.8] {
        for &b in &[0.2, 0.5, 0.9] {
            assert_eq!(tconorm(small, a, b), tconorm(maximum, a, b));
            assert_eq!(tconorm(large, a, b), tconorm(bounded, a, b));
        }
    }
}

#[test]
fn dombi_p1_equals_hamacher() {
    let dombi = TConormFamily::Dombi { p: 1.0 };
    let hamacher = TConormFamily::Hamacher;
    for &a in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        for &b in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let actual = tconorm(dombi, a, b);
            let expected = tconorm(hamacher, a, b);
            assert!(
                (actual - expected).abs() < 1e-12,
                "Dombi(p=1) != Hamacher for S({a},{b}): {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn recovered_catalog_limit_cases_match_basic_families() {
    let grid = [0.0, 0.2, 0.5, 0.8, 1.0];
    for &a in &grid {
        for &b in &grid {
            assert!(
                (tnorm(TConormFamily::SchweizerSklar { p: 0.0 }, a, b) - tnorm(PRODUCT, a, b))
                    .abs()
                    < 1e-12
            );
            assert!(
                (tnorm(TConormFamily::SchweizerSklar { p: 1.0 }, a, b) - tnorm(LUKASIEWICZ, a, b))
                    .abs()
                    < 1e-12
            );
            assert!(
                (tnorm(TConormFamily::SugenoWeber { p: 0.0 }, a, b) - tnorm(LUKASIEWICZ, a, b))
                    .abs()
                    < 1e-12
            );
            assert!(
                (tnorm(TConormFamily::AczelAlsina { p: 1.0 }, a, b) - tnorm(PRODUCT, a, b)).abs()
                    < 1e-12
            );
            assert!(
                (tnorm(TConormFamily::MayorTorrens { p: 0.0 }, a, b) - tnorm(GODEL, a, b)).abs()
                    < 1e-12
            );
            assert!(
                (tnorm(TConormFamily::MayorTorrens { p: 1.0 }, a, b) - tnorm(LUKASIEWICZ, a, b))
                    .abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn copula_metadata_is_conservative() {
    assert_eq!(TConormFamily::Maximum.is_copula(), Some(true));
    assert_eq!(TConormFamily::Probabilistic.is_copula(), Some(true));
    assert_eq!(TConormFamily::Bounded.is_copula(), Some(true));
    assert_eq!(TConormFamily::Frank { s: 2.0 }.is_copula(), Some(true));
    assert_eq!(TConormFamily::Drastic.is_copula(), Some(false));
    assert_eq!(TConormFamily::NilpotentMinimum.is_copula(), Some(false));
    assert_eq!(TConormFamily::Dombi { p: 0.5 }.is_copula(), None);
    assert_eq!(
        TConormFamily::Dombi { p: 0.5 }.copula_compatibility(),
        CopulaCompatibility::ParameterDependent
    );
    assert_eq!(
        TConormFamily::Yager { p: 2.0 }.copula_compatibility(),
        CopulaCompatibility::Unknown
    );
}

#[test]
fn generic_float_path_matches_f64_for_representative_families() {
    let a = 0.375_f32;
    let b = 0.625_f32;
    for family in all_families() {
        let f32_value = tconorm_generic(family, a, b);
        let f64_value = tconorm(family, f64::from(a), f64::from(b)) as f32;
        assert!(
            (f32_value - f64_value).abs() < 1e-5,
            "{family:?}: generic f32 path {f32_value} != f64 path {f64_value}"
        );

        let f32_value = tnorm_generic(family, a, b);
        let f64_value = tnorm(family, f64::from(a), f64::from(b)) as f32;
        assert!(
            (f32_value - f64_value).abs() < 1e-5,
            "{family:?}: generic f32 tnorm path {f32_value} != f64 path {f64_value}"
        );
    }
}

#[test]
fn log_product_matches_product_fold_without_underflow() {
    let values = [0.9, 0.8, 0.7, 0.6];
    let log_value = log_product(&values, 1e-12);
    assert!((product_from_log(log_value) - tnorm_fold(PRODUCT, &values)).abs() < 1e-12);
    assert_eq!(log_product(&[], 1e-12), 0.0);
}

#[test]
fn log_product_floor_handles_zero_values() {
    let log_value = log_product(&[1.0, 0.0, 0.5], 1e-6);
    assert!((product_from_log(log_value) - 5e-7).abs() < 1e-18);
    assert_eq!(product_from_log(f64::NEG_INFINITY), 0.0);
    assert_eq!(product_from_log(f64::INFINITY), 1.0);
    assert_eq!(product_from_log(f64::NAN), 0.0);
}

#[test]
fn generators_reconstruct_tnorm_and_residuum() {
    let families = [
        TConormFamily::Probabilistic,
        TConormFamily::Bounded,
        TConormFamily::Einstein,
        TConormFamily::Hamacher,
        TConormFamily::Yager { p: 2.0 },
        TConormFamily::Frank { s: 2.0 },
        TConormFamily::Dombi { p: 2.0 },
        TConormFamily::SchweizerSklar { p: -1.0 },
        TConormFamily::SchweizerSklar { p: 0.5 },
        TConormFamily::SugenoWeber { p: 0.5 },
        TConormFamily::AczelAlsina { p: 2.0 },
    ];

    for family in families {
        let generator = family.generator().expect("family should have generator");
        for &(a, b) in &[(0.25, 0.8), (0.5, 0.6), (0.8, 0.4)] {
            assert!(
                (generator.tnorm(a, b) - tnorm(family, a, b)).abs() < 1e-10,
                "{family:?}: generator t-norm mismatch for {a}, {b}"
            );
            let implication = family.residuum(a, b).unwrap();
            let left = tnorm(family, a, implication);
            assert!(
                left <= b + 1e-10 || (implication - 1.0).abs() < 1e-12,
                "{family:?}: residuum {implication} gave T({a},I)={left} > {b}"
            );
        }
    }
}

#[test]
fn direct_residua_cover_non_generator_left_continuous_families() {
    assert_eq!(residuum(TConormFamily::Drastic, 0.8, 0.4), None);
    assert!((residuum(TConormFamily::Maximum, 0.8, 0.4).unwrap() - 0.4).abs() < 1e-12);
    assert!((residuum(TConormFamily::NilpotentMinimum, 0.8, 0.4).unwrap() - 0.4).abs() < 1e-12);
    let mayor = TConormFamily::MayorTorrens { p: 0.4 };
    let implication = residuum(mayor, 0.8, 0.4).unwrap();
    assert!(tnorm(mayor, 0.8, implication) <= 0.4 + 1e-10);
}

#[test]
fn negation_and_implication_families_match_reference_formulas() {
    assert!((negation(NegationFamily::Standard, 0.25) - 0.75).abs() < 1e-12);
    assert!(
        (negation(NegationFamily::Sugeno { lambda: 0.5 }, 0.25) - 0.6666666666666666).abs() < 1e-12
    );
    assert!((negation(NegationFamily::Yager { p: 2.0 }, 0.6) - 0.8).abs() < 1e-12);

    assert!((implication(ImplicationFamily::KleeneDienes, 0.8, 0.3) - 0.3).abs() < 1e-12);
    assert!((implication(ImplicationFamily::Reichenbach, 0.8, 0.3) - 0.44).abs() < 1e-12);
    assert!((implication(ImplicationFamily::Lukasiewicz, 0.8, 0.3) - 0.5).abs() < 1e-12);
    assert_eq!(
        implication(ImplicationFamily::Godel, 0.8, 0.3),
        LogicFamily::Godel.residuum(0.8, 0.3)
    );
    assert_eq!(
        implication(ImplicationFamily::Product, 0.8, 0.3),
        LogicFamily::Product.residuum(0.8, 0.3)
    );
}

#[test]
fn aggregation_operators_cover_owa_power_mean_and_uninorm() {
    let mut values = [0.2, 0.9, 0.4];
    let result = owa(&[0.5, 0.3, 0.2], &mut values).unwrap();
    assert!((result - 0.61).abs() < 1e-12);
    assert_eq!(values, [0.9, 0.4, 0.2]);

    assert!((power_mean(&[0.25, 1.0], 1.0).unwrap() - 0.625).abs() < 1e-12);
    assert!((power_mean(&[0.25, 1.0], f64::INFINITY).unwrap() - 1.0).abs() < 1e-12);
    assert!((power_mean(&[0.25, 1.0], f64::NEG_INFINITY).unwrap() - 0.25).abs() < 1e-12);
    assert!(
        (power_mean_weighted(&[0.25, 1.0], 1.0, |i| if i == 0 { 0.25 } else { 0.75 }).unwrap()
            - 0.8125)
            .abs()
            < 1e-12
    );

    let e = 0.4;
    assert!(
        (representable_uninorm(PRODUCT, PRODUCT, MixedRegion::Min, e, e, 0.8) - 0.8).abs() < 1e-12
    );
    assert!(
        (representable_uninorm(PRODUCT, PRODUCT, MixedRegion::Min, e, 0.2, e) - 0.2).abs() < 1e-12
    );
    assert!(
        (representable_uninorm(PRODUCT, PRODUCT, MixedRegion::Min, e, 0.2, 0.3)
            - e * tnorm(PRODUCT, 0.5, 0.75))
        .abs()
            < 1e-12
    );
}

#[test]
fn ordinal_sums_use_segment_formula_and_min_max_fallbacks() {
    let segments = [
        OrdinalSegment::new(0.2, 0.6, PRODUCT),
        OrdinalSegment::new(0.7, 0.9, LUKASIEWICZ),
    ];
    let sum = OrdinalSum::new(&segments).unwrap();

    let tnorm_value = sum.tnorm(0.4, 0.5);
    assert!((tnorm_value - (0.2 + 0.4 * tnorm(PRODUCT, 0.5, 0.75))).abs() < 1e-12);

    let tconorm_value = sum.tconorm(0.4, 0.5);
    assert!((tconorm_value - (0.2 + 0.4 * tconorm(PRODUCT, 0.5, 0.75))).abs() < 1e-12);

    assert!((sum.tnorm(0.4, 0.8) - 0.4).abs() < 1e-12);
    assert!((sum.tconorm(0.4, 0.8) - 0.8).abs() < 1e-12);
    assert_eq!(sum.segments(), &segments);
}

#[test]
fn ordinal_sums_reject_invalid_or_overlapping_segments() {
    let inverted = [OrdinalSegment::new(0.6, 0.2, PRODUCT)];
    assert!(OrdinalSum::new(&inverted).is_none());

    let overlapping = [
        OrdinalSegment::new(0.2, 0.6, PRODUCT),
        OrdinalSegment::new(0.5, 0.8, LUKASIEWICZ),
    ];
    assert!(OrdinalSum::new(&overlapping).is_none());
}

#[test]
fn tnorm_duality() {
    // T(a,b) = 1 - S(1-a, 1-b)
    // For probabilistic: T(a,b) = ab
    let f = TConormFamily::Probabilistic;
    let t = tnorm(f, 0.5, 0.6);
    assert!((t - 0.3).abs() < 1e-10, "T(0.5,0.6) = {t}, expected 0.3");
}

#[test]
fn catalog_gradients_match_finite_differences_away_from_kinks() {
    for (family, a, b) in [
        (TConormFamily::Probabilistic, 0.4, 0.7),
        (TConormFamily::Einstein, 0.4, 0.7),
        (TConormFamily::Hamacher, 0.4, 0.7),
        (TConormFamily::Yager { p: 2.0 }, 0.8, 0.9),
        (TConormFamily::Frank { s: 2.0 }, 0.4, 0.7),
        (TConormFamily::Dombi { p: 2.0 }, 0.4, 0.7),
        (TConormFamily::SchweizerSklar { p: -1.0 }, 0.4, 0.7),
        (TConormFamily::SchweizerSklar { p: 0.5 }, 0.8, 0.9),
        (TConormFamily::SugenoWeber { p: 0.5 }, 0.8, 0.9),
        (TConormFamily::AczelAlsina { p: 2.0 }, 0.4, 0.7),
        (TConormFamily::MayorTorrens { p: 0.4 }, 0.1, 0.2),
    ] {
        assert_matches_finite_difference(
            tnorm_gradient(family, a, b),
            finite_difference(|x, y| tnorm(family, x, y), a, b),
        );
        assert_matches_finite_difference(
            tconorm_gradient(family, a, b),
            finite_difference(|x, y| tconorm(family, x, y), a, b),
        );
    }
}

#[test]
fn named_family_constants_match_common_tnorms() {
    assert!((tnorm(GODEL, 0.4, 0.7) - 0.4).abs() < 1e-12);
    assert!((tconorm(GODEL, 0.4, 0.7) - 0.7).abs() < 1e-12);

    assert!((tnorm(PRODUCT, 0.4, 0.7) - 0.28).abs() < 1e-12);
    assert!((tconorm(PRODUCT, 0.4, 0.7) - 0.82).abs() < 1e-12);

    assert!((tnorm(LUKASIEWICZ, 0.4, 0.7) - 0.1).abs() < 1e-12);
    assert!((tconorm(LUKASIEWICZ, 0.4, 0.7) - 1.0).abs() < 1e-12);
}

#[test]
fn logic_family_residua_match_common_formulas() {
    assert!((LogicFamily::Godel.residuum(0.4, 0.7) - 1.0).abs() < 1e-12);
    assert!((LogicFamily::Godel.residuum(0.7, 0.4) - 0.4).abs() < 1e-12);

    assert!((LogicFamily::Product.residuum(0.4, 0.7) - 1.0).abs() < 1e-12);
    assert!((LogicFamily::Product.residuum(0.7, 0.28) - 0.4).abs() < 1e-12);

    assert!((LogicFamily::Lukasiewicz.residuum(0.7, 0.4) - 0.7).abs() < 1e-12);
    assert!((LogicFamily::Lukasiewicz.neg(0.4) - 0.6).abs() < 1e-12);
}

fn assert_gradient_close(actual: BinaryGradient<f64>, expected: BinaryGradient<f64>) {
    assert!(
        (actual.da - expected.da).abs() < 1e-10,
        "da: actual {}, expected {}",
        actual.da,
        expected.da
    );
    assert!(
        (actual.db - expected.db).abs() < 1e-10,
        "db: actual {}, expected {}",
        actual.db,
        expected.db
    );
}

fn finite_difference<F>(op: F, a: f64, b: f64) -> BinaryGradient<f64>
where
    F: Fn(f64, f64) -> f64,
{
    let h = 1e-6;
    BinaryGradient::new(
        (op(a + h, b) - op(a - h, b)) / (2.0 * h),
        (op(a, b + h) - op(a, b - h)) / (2.0 * h),
    )
}

fn assert_matches_finite_difference(actual: BinaryGradient<f64>, numeric: BinaryGradient<f64>) {
    assert!(
        (actual.da - numeric.da).abs() < 1e-6,
        "da: actual {}, numeric {}",
        actual.da,
        numeric.da
    );
    assert!(
        (actual.db - numeric.db).abs() < 1e-6,
        "db: actual {}, numeric {}",
        actual.db,
        numeric.db
    );
}

#[test]
fn logic_family_gradients_match_core_formulas() {
    assert_gradient_close(
        LogicFamily::Godel.tnorm_gradient(0.3, 0.7),
        BinaryGradient::new(1.0, 0.0),
    );
    assert_gradient_close(
        LogicFamily::Godel.tconorm_gradient(0.3, 0.7),
        BinaryGradient::new(0.0, 1.0),
    );
    assert_gradient_close(
        LogicFamily::Godel.residuum_gradient(0.7, 0.4),
        BinaryGradient::new(0.0, 1.0),
    );

    assert_gradient_close(
        LogicFamily::Product.tnorm_gradient(0.4, 0.7),
        BinaryGradient::new(0.7, 0.4),
    );
    assert_gradient_close(
        LogicFamily::Product.tconorm_gradient(0.4, 0.7),
        BinaryGradient::new(0.3, 0.6),
    );
    assert_gradient_close(
        LogicFamily::Product.residuum_gradient(0.8, 0.4),
        BinaryGradient::new(-0.625, 1.25),
    );

    assert_gradient_close(
        LogicFamily::Lukasiewicz.tnorm_gradient(0.7, 0.6),
        BinaryGradient::new(1.0, 1.0),
    );
    assert_gradient_close(
        LogicFamily::Lukasiewicz.tconorm_gradient(0.2, 0.3),
        BinaryGradient::new(1.0, 1.0),
    );
    assert_gradient_close(
        LogicFamily::Lukasiewicz.residuum_gradient(0.7, 0.4),
        BinaryGradient::new(-1.0, 1.0),
    );
}

#[test]
fn logic_family_gradients_match_finite_differences_away_from_kinks() {
    for (logic, a, b) in [
        (LogicFamily::Godel, 0.3, 0.7),
        (LogicFamily::Product, 0.4, 0.7),
        (LogicFamily::Lukasiewicz, 0.7, 0.6),
    ] {
        assert_matches_finite_difference(
            logic.tnorm_gradient(a, b),
            finite_difference(|x, y| logic.tnorm(x, y), a, b),
        );
    }

    for (logic, a, b) in [
        (LogicFamily::Godel, 0.3, 0.7),
        (LogicFamily::Product, 0.4, 0.7),
        (LogicFamily::Lukasiewicz, 0.2, 0.3),
    ] {
        assert_matches_finite_difference(
            logic.tconorm_gradient(a, b),
            finite_difference(|x, y| logic.tconorm(x, y), a, b),
        );
    }

    for (logic, a, b) in [
        (LogicFamily::Godel, 0.7, 0.4),
        (LogicFamily::Product, 0.8, 0.4),
        (LogicFamily::Lukasiewicz, 0.7, 0.4),
    ] {
        assert_matches_finite_difference(
            logic.residuum_gradient(a, b),
            finite_difference(|x, y| logic.residuum(x, y), a, b),
        );
    }
}

#[test]
fn logic_family_gradient_kink_conventions_are_pinned() {
    assert_gradient_close(
        LogicFamily::Godel.tnorm_gradient(0.5, 0.5),
        BinaryGradient::new(0.5, 0.5),
    );
    assert_gradient_close(
        LogicFamily::Godel.tconorm_gradient(0.5, 0.5),
        BinaryGradient::new(0.5, 0.5),
    );
    assert_gradient_close(
        LogicFamily::Lukasiewicz.tnorm_gradient(0.4, 0.6),
        BinaryGradient::new(0.0, 0.0),
    );
    assert_gradient_close(
        LogicFamily::Lukasiewicz.tconorm_gradient(0.4, 0.6),
        BinaryGradient::new(0.0, 0.0),
    );
    assert_gradient_close(
        LogicFamily::Lukasiewicz.residuum_gradient(0.5, 0.5),
        BinaryGradient::new(0.0, 0.0),
    );
}

#[test]
fn logic_family_f32_gradients_match_f64() {
    let a = 0.4_f32;
    let b = 0.7_f32;
    for logic in [
        LogicFamily::Godel,
        LogicFamily::Product,
        LogicFamily::Lukasiewicz,
    ] {
        let tnorm = logic.tnorm_gradient_f32(a, b);
        let tnorm_f64 = logic.tnorm_gradient(f64::from(a), f64::from(b));
        assert!((tnorm.da - tnorm_f64.da as f32).abs() < 1e-6);
        assert!((tnorm.db - tnorm_f64.db as f32).abs() < 1e-6);

        let tconorm = logic.tconorm_gradient_f32(a, b);
        let tconorm_f64 = logic.tconorm_gradient(f64::from(a), f64::from(b));
        assert!((tconorm.da - tconorm_f64.da as f32).abs() < 1e-6);
        assert!((tconorm.db - tconorm_f64.db as f32).abs() < 1e-6);

        let residuum = logic.residuum_gradient_f32(a, b);
        let residuum_f64 = logic.residuum_gradient(f64::from(a), f64::from(b));
        assert!((residuum.da - residuum_f64.da as f32).abs() < 1e-6);
        assert!((residuum.db - residuum_f64.db as f32).abs() < 1e-6);
        assert!((logic.neg_gradient_f32(a) - logic.neg_gradient(f64::from(a)) as f32).abs() < 1e-6);
    }
}

#[test]
fn logic_family_residuation_law_holds_on_grid() {
    let grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    for logic in [
        LogicFamily::Godel,
        LogicFamily::Product,
        LogicFamily::Lukasiewicz,
    ] {
        for &a in &grid {
            for &b in &grid {
                for &c in &grid {
                    let left = logic.tnorm(a, c) <= b + 1e-12;
                    let right = c <= logic.residuum(a, b) + 1e-12;
                    assert_eq!(
                        left, right,
                        "{logic:?}: {a} * {c} <= {b} iff {c} <= ({a} -> {b})"
                    );
                }
            }
        }
    }
}

#[test]
fn logic_family_f32_helpers_match_f64() {
    let a = 0.4_f32;
    let b = 0.7_f32;
    for logic in [
        LogicFamily::Godel,
        LogicFamily::Product,
        LogicFamily::Lukasiewicz,
    ] {
        assert!(
            (logic.tnorm_f32(a, b) - logic.tnorm(f64::from(a), f64::from(b)) as f32).abs() < 1e-6
        );
        assert!(
            (logic.tconorm_f32(a, b) - logic.tconorm(f64::from(a), f64::from(b)) as f32).abs()
                < 1e-6
        );
        assert!(
            (logic.residuum_f32(a, b) - logic.residuum(f64::from(a), f64::from(b)) as f32).abs()
                < 1e-6
        );
    }
}

#[test]
fn logic_family_f32_helpers_clamp_like_f64() {
    let values = [-0.25_f32, 0.0, 0.4, 1.0, 1.5];
    for logic in [
        LogicFamily::Godel,
        LogicFamily::Product,
        LogicFamily::Lukasiewicz,
    ] {
        for &a in &values {
            for &b in &values {
                assert!(
                    (logic.tnorm_f32(a, b) - logic.tnorm(f64::from(a), f64::from(b)) as f32).abs()
                        < 1e-6
                );
                assert!(
                    (logic.tconorm_f32(a, b) - logic.tconorm(f64::from(a), f64::from(b)) as f32)
                        .abs()
                        < 1e-6
                );
                assert!(
                    (logic.residuum_f32(a, b) - logic.residuum(f64::from(a), f64::from(b)) as f32)
                        .abs()
                        < 1e-6
                );
            }
        }
    }
}

#[test]
fn fold_empty_returns_identity() {
    let f = TConormFamily::Maximum;
    assert_eq!(tconorm_fold(f, &[]), 0.0);
    assert_eq!(tnorm_fold(f, &[]), 1.0);
}

#[test]
fn fold_single_value() {
    let f = TConormFamily::Probabilistic;
    assert!((tconorm_fold(f, &[0.7]) - 0.7).abs() < 1e-10);
}

#[test]
fn fold_associativity() {
    let f = TConormFamily::Einstein;
    let abc = tconorm_fold(f, &[0.3, 0.5, 0.7]);
    let ab_c = tconorm(f, tconorm(f, 0.3, 0.5), 0.7);
    assert!(
        (abc - ab_c).abs() < 1e-10,
        "fold should match left-to-right application"
    );
}

fn degree() -> impl Strategy<Value = f32> {
    (0.0_f32..=1.0).prop_map(|x| (x * 1000.0).round() / 1000.0)
}

#[test]
fn catalog_laws_hold_on_sobol_grid() {
    for point in lowdisc::sobol_sequence(128, 3) {
        let a = point[0];
        let b = point[1];
        let c = point[2];
        for family in all_families() {
            assert!((tconorm(family, a, b) - tconorm(family, b, a)).abs() < 1e-9);
            assert!((tnorm(family, a, b) - tnorm(family, b, a)).abs() < 1e-9);

            let s_left = tconorm(family, tconorm(family, a, b), c);
            let s_right = tconorm(family, a, tconorm(family, b, c));
            assert!(
                (s_left - s_right).abs() < 1e-8,
                "{family:?}: t-conorm associativity failed at {a}, {b}, {c}"
            );

            let t_left = tnorm(family, tnorm(family, a, b), c);
            let t_right = tnorm(family, a, tnorm(family, b, c));
            assert!(
                (t_left - t_right).abs() < 1e-8,
                "{family:?}: t-norm associativity failed at {a}, {b}, {c}"
            );
        }
    }
}

fn check_algebra_matches_logic<A: TruthAlgebra>(logic: LogicFamily, a: f32, b: f32) {
    assert!((A::tnorm(a.into(), b.into()) - logic.tnorm(a.into(), b.into())).abs() < 1e-12);
    assert!((A::tconorm(a.into(), b.into()) - logic.tconorm(a.into(), b.into())).abs() < 1e-12);
    assert!((A::residuum(a.into(), b.into()) - logic.residuum(a.into(), b.into())).abs() < 1e-12);
    assert!((A::neg(a.into()) - logic.neg(a.into())).abs() < 1e-12);
    assert_eq!(
        A::tnorm_gradient(a.into(), b.into()),
        logic.tnorm_gradient(a.into(), b.into())
    );
    assert_eq!(
        A::tconorm_gradient(a.into(), b.into()),
        logic.tconorm_gradient(a.into(), b.into())
    );
    assert_eq!(
        A::residuum_gradient(a.into(), b.into()),
        logic.residuum_gradient(a.into(), b.into())
    );
    assert_eq!(A::neg_gradient(a.into()), logic.neg_gradient(a.into()));
    assert!((A::tnorm_f32(a, b) - logic.tnorm_f32(a, b)).abs() < 1e-6);
    assert!((A::tconorm_f32(a, b) - logic.tconorm_f32(a, b)).abs() < 1e-6);
    assert!((A::residuum_f32(a, b) - logic.residuum_f32(a, b)).abs() < 1e-6);
    assert!((A::neg_f32(a) - logic.neg_f32(a)).abs() < 1e-6);
    assert_eq!(A::tnorm_gradient_f32(a, b), logic.tnorm_gradient_f32(a, b));
    assert_eq!(
        A::tconorm_gradient_f32(a, b),
        logic.tconorm_gradient_f32(a, b)
    );
    assert_eq!(
        A::residuum_gradient_f32(a, b),
        logic.residuum_gradient_f32(a, b)
    );
    assert_eq!(A::neg_gradient_f32(a), logic.neg_gradient_f32(a));
}

#[test]
fn truth_algebra_markers_match_logic_family() {
    check_algebra_matches_logic::<Godel>(LogicFamily::Godel, 0.7, 0.4);
    check_algebra_matches_logic::<Product>(LogicFamily::Product, 0.7, 0.4);
    check_algebra_matches_logic::<Lukasiewicz>(LogicFamily::Lukasiewicz, 0.7, 0.4);
}

fn check_truth_algebra_laws<A: TruthAlgebra>(a: f32, b: f32, c: f32) -> Result<(), TestCaseError> {
    const EPS: f32 = 1e-5;

    prop_assert!((A::tnorm_f32(a, A::top_f32()) - a).abs() < EPS);
    prop_assert!((A::tnorm_f32(a, A::bottom_f32()) - A::bottom_f32()).abs() < EPS);
    prop_assert!((A::tconorm_f32(a, A::bottom_f32()) - a).abs() < EPS);
    prop_assert!((A::tconorm_f32(a, A::top_f32()) - A::top_f32()).abs() < EPS);

    prop_assert!((A::tnorm_f32(a, b) - A::tnorm_f32(b, a)).abs() < EPS);
    prop_assert!((A::tconorm_f32(a, b) - A::tconorm_f32(b, a)).abs() < EPS);

    prop_assert!(
        (A::tnorm_f32(A::tnorm_f32(a, b), c) - A::tnorm_f32(a, A::tnorm_f32(b, c))).abs() < EPS
    );
    prop_assert!(
        (A::tconorm_f32(A::tconorm_f32(a, b), c) - A::tconorm_f32(a, A::tconorm_f32(b, c))).abs()
            < EPS
    );

    prop_assert!(A::tnorm_f32(a, A::residuum_f32(a, b)) <= b + EPS);
    prop_assert!(c <= A::residuum_f32(a, A::tnorm_f32(a, c)) + EPS);
    if a <= b {
        prop_assert!((A::residuum_f32(a, b) - A::top_f32()).abs() < EPS);
    }

    Ok(())
}

proptest! {
    #[test]
    fn catalog_tconorm_laws_hold(a in degree(), b in degree(), c in degree(), d in degree()) {
        let a = f64::from(a);
        let b = f64::from(b);
        let c = f64::from(c);
        let lo = a.min(d.into());
        let hi = a.max(d.into());

        for family in all_families() {
            let ab = tconorm(family, a, b);
            prop_assert!(ab.is_finite(), "{family:?}: S({a},{b}) = {ab}");
            prop_assert!((-1e-12..=1.0 + 1e-12).contains(&ab));
            prop_assert!(
                (ab - tconorm(family, b, a)).abs() < 1e-9,
                "{family:?}: commutativity failed for {a}, {b}"
            );
            prop_assert!(
                (tconorm(family, a, 0.0) - a).abs() < 1e-9,
                "{family:?}: zero identity failed for {a}"
            );
            prop_assert!(
                (tconorm(family, a, 1.0) - 1.0).abs() < 1e-9,
                "{family:?}: one absorption failed for {a}"
            );
            prop_assert!(
                tconorm(family, lo, b) <= tconorm(family, hi, b) + 1e-9,
                "{family:?}: monotonicity failed for {lo} <= {hi}, {b}"
            );

            let left = tconorm(family, tconorm(family, a, b), c);
            let right = tconorm(family, a, tconorm(family, b, c));
            prop_assert!(
                (left - right).abs() < 1e-8,
                "{family:?}: associativity failed for {a}, {b}, {c}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn catalog_tnorm_laws_hold(a in degree(), b in degree(), c in degree(), d in degree()) {
        let a = f64::from(a);
        let b = f64::from(b);
        let c = f64::from(c);
        let lo = a.min(d.into());
        let hi = a.max(d.into());

        for family in all_families() {
            let ab = tnorm(family, a, b);
            prop_assert!(ab.is_finite(), "{family:?}: T({a},{b}) = {ab}");
            prop_assert!((-1e-12..=1.0 + 1e-12).contains(&ab));
            prop_assert!(
                (ab - tnorm(family, b, a)).abs() < 1e-9,
                "{family:?}: commutativity failed for {a}, {b}"
            );
            prop_assert!(
                (tnorm(family, a, 1.0) - a).abs() < 1e-9,
                "{family:?}: one identity failed for {a}"
            );
            prop_assert!(
                tnorm(family, a, 0.0).abs() < 1e-9,
                "{family:?}: zero absorption failed for {a}"
            );
            prop_assert!(
                tnorm(family, lo, b) <= tnorm(family, hi, b) + 1e-9,
                "{family:?}: monotonicity failed for {lo} <= {hi}, {b}"
            );

            let left = tnorm(family, tnorm(family, a, b), c);
            let right = tnorm(family, a, tnorm(family, b, c));
            prop_assert!(
                (left - right).abs() < 1e-8,
                "{family:?}: associativity failed for {a}, {b}, {c}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn parametric_tconorms_remain_finite_and_bounded(a in degree(), b in degree()) {
        for family in [
            TConormFamily::Yager { p: 128.0 },
            TConormFamily::Frank { s: 1.0 },
            TConormFamily::Frank { s: 1.0 + 1e-9 },
            TConormFamily::Frank { s: 2.0 },
            TConormFamily::Frank {
                s: f64::MIN_POSITIVE,
            },
            TConormFamily::Frank { s: f64::MAX },
            TConormFamily::Dombi { p: 1.0 },
            TConormFamily::Dombi { p: 64.0 },
        ] {
            let value = tconorm(family, f64::from(a), f64::from(b));
            prop_assert!(value.is_finite(), "{family:?}: S({a},{b}) = {value}");
            prop_assert!(
                (-1e-12..=1.0 + 1e-12).contains(&value),
                "{family:?}: S({a},{b}) = {value}"
            );
        }
    }

    #[test]
    fn godel_truth_algebra_laws_hold(a in degree(), b in degree(), c in degree()) {
        check_truth_algebra_laws::<Godel>(a, b, c)?;
    }

    #[test]
    fn product_truth_algebra_laws_hold(a in degree(), b in degree(), c in degree()) {
        check_truth_algebra_laws::<Product>(a, b, c)?;
    }

    #[test]
    fn lukasiewicz_truth_algebra_laws_hold(a in degree(), b in degree(), c in degree()) {
        check_truth_algebra_laws::<Lukasiewicz>(a, b, c)?;
    }
}

#[cfg(feature = "burn-ndarray")]
mod burn_tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    type B = burn_ndarray::NdArray;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn tensor(values: [f32; 4]) -> Tensor<B, 2> {
        Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), [2, 2]), &device())
    }

    fn values(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor.into_data().to_vec::<f32>().unwrap()
    }

    fn assert_tensor_close(actual: Tensor<B, 2>, expected: Vec<f32>) {
        let actual = values(actual);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn burn_tensor_ops_match_scalar_logic_families() {
        let a_values = [0.2, 0.4, 0.7, 1.2];
        let b_values = [0.3, 0.8, 0.6, -0.2];

        for logic in [
            LogicFamily::Godel,
            LogicFamily::Product,
            LogicFamily::Lukasiewicz,
        ] {
            assert_tensor_close(
                logic.tnorm_tensor(tensor(a_values), tensor(b_values)),
                a_values
                    .iter()
                    .zip(b_values.iter())
                    .map(|(&a, &b)| logic.tnorm_f32(a, b))
                    .collect(),
            );
            assert_tensor_close(
                logic.tconorm_tensor(tensor(a_values), tensor(b_values)),
                a_values
                    .iter()
                    .zip(b_values.iter())
                    .map(|(&a, &b)| logic.tconorm_f32(a, b))
                    .collect(),
            );
            assert_tensor_close(
                logic.residuum_tensor(tensor(a_values), tensor(b_values)),
                a_values
                    .iter()
                    .zip(b_values.iter())
                    .map(|(&a, &b)| logic.residuum_f32(a, b))
                    .collect(),
            );
            assert_tensor_close(
                logic.neg_tensor(tensor(a_values)),
                a_values.iter().map(|&a| logic.neg_f32(a)).collect(),
            );
        }
    }

    #[test]
    fn truth_algebra_burn_tensor_ops_delegate_to_logic_family() {
        let a = tensor([0.2, 0.4, 0.7, 0.9]);
        let b = tensor([0.3, 0.8, 0.6, 0.1]);
        assert_tensor_close(
            Product::tnorm_tensor(a.clone(), b.clone()),
            values(LogicFamily::Product.tnorm_tensor(a.clone(), b.clone())),
        );
        assert_tensor_close(
            Product::tconorm_tensor(a.clone(), b.clone()),
            values(LogicFamily::Product.tconorm_tensor(a.clone(), b.clone())),
        );
        assert_tensor_close(
            Product::residuum_tensor(a.clone(), b.clone()),
            values(LogicFamily::Product.residuum_tensor(a.clone(), b.clone())),
        );
        assert_tensor_close(
            Product::neg_tensor(a.clone()),
            values(LogicFamily::Product.neg_tensor(a)),
        );
    }

    #[test]
    fn burn_product_tnorm_records_autodiff_gradients() {
        type AD = burn::backend::Autodiff<burn_ndarray::NdArray>;

        let device = burn_ndarray::NdArrayDevice::Cpu;
        let a = Tensor::<AD, 1>::from_data(TensorData::new(vec![0.4, 0.6], [2]), &device)
            .require_grad();
        let b = Tensor::<AD, 1>::from_data(TensorData::new(vec![0.7, 0.2], [2]), &device)
            .require_grad();

        let loss = LogicFamily::Product
            .tnorm_tensor(a.clone(), b.clone())
            .sum();
        let grads = loss.backward();
        let grad_a = values_1d(a.grad(&grads).unwrap());
        let grad_b = values_1d(b.grad(&grads).unwrap());

        assert_eq!(grad_a, vec![0.7, 0.2]);
        assert_eq!(grad_b, vec![0.4, 0.6]);
    }

    fn values_1d(tensor: Tensor<burn_ndarray::NdArray, 1>) -> Vec<f32> {
        tensor.into_data().to_vec::<f32>().unwrap()
    }
}
