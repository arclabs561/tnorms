//! Adjunction and negation-involution laws for the primary t-norm surface.
//!
//! In a residuated lattice the t-norm is primary and its residuum is the right
//! adjoint: `a ⊗ b <= c` if and only if `a <= b → c`. The t-conorm is
//! secondary, obtained by duality.

use tnorms::{Godel, LogicFamily, Lukasiewicz, Product, TNorm};

const TOL: f64 = 1e-12;

/// Decimal steps plus values that are not exact binary fractions, so the
/// branch comparisons inside each residuum are exercised away from ties.
const GRID: [f64; 12] = [
    0.0,
    0.05,
    0.1,
    1.0 / 3.0,
    0.4,
    0.5,
    0.6,
    0.7,
    2.0 / 3.0,
    0.9,
    0.95,
    1.0,
];

fn assert_adjunction<T: TNorm>() {
    for &a in &GRID {
        for &b in &GRID {
            for &c in &GRID {
                let product_le_c = T::tnorm(a, b) <= c + TOL;
                let a_le_implication = a <= T::residuum(b, c) + TOL;
                assert_eq!(
                    product_le_c,
                    a_le_implication,
                    "{}: a = {a}, b = {b}, c = {c}: a⊗b = {}, b→c = {}",
                    T::NAME,
                    T::tnorm(a, b),
                    T::residuum(b, c)
                );
            }
        }
    }
}

#[test]
fn godel_is_an_adjoint_pair() {
    assert_adjunction::<Godel>();
}

#[test]
fn product_is_an_adjoint_pair() {
    assert_adjunction::<Product>();
}

#[test]
fn lukasiewicz_is_an_adjoint_pair() {
    assert_adjunction::<Lukasiewicz>();
}

#[test]
fn only_lukasiewicz_residual_negation_is_involutive() {
    for &a in &GRID {
        let twice = Lukasiewicz::negation(Lukasiewicz::negation(a));
        assert!((twice - a).abs() <= TOL, "lukasiewicz: ¬¬{a} = {twice}");
    }

    for (name, twice) in [
        ("godel", Godel::negation(Godel::negation(0.5))),
        ("product", Product::negation(Product::negation(0.5))),
    ] {
        assert!(
            (twice - 0.5).abs() > 1e-3,
            "{name}: residual negation must not be involutive, got ¬¬0.5 = {twice}"
        );
    }
}

fn assert_delegates_bitwise<T: TNorm>(lf: LogicFamily) {
    for &a in &GRID {
        for &b in &GRID {
            assert_eq!(
                T::tnorm(a, b).to_bits(),
                lf.tnorm(a, b).to_bits(),
                "{}: tnorm({a}, {b})",
                T::NAME
            );
            assert_eq!(
                T::residuum(a, b).to_bits(),
                lf.residuum(a, b).to_bits(),
                "{}: residuum({a}, {b})",
                T::NAME
            );
            assert_eq!(
                T::tconorm(a, b).to_bits(),
                lf.tconorm(a, b).to_bits(),
                "{}: tconorm({a}, {b})",
                T::NAME
            );
        }
        assert_eq!(
            T::negation(a).to_bits(),
            lf.neg(a).to_bits(),
            "{}: negation({a})",
            T::NAME
        );
    }
}

#[test]
fn markers_delegate_to_logic_family_bit_for_bit() {
    assert_delegates_bitwise::<Godel>(LogicFamily::Godel);
    assert_delegates_bitwise::<Product>(LogicFamily::Product);
    assert_delegates_bitwise::<Lukasiewicz>(LogicFamily::Lukasiewicz);
}

#[test]
fn tconorm_is_the_standard_negation_dual_of_the_tnorm() {
    fn assert_duality<T: TNorm>() {
        for &a in &GRID {
            for &b in &GRID {
                let dual = 1.0 - T::tnorm(1.0 - a, 1.0 - b);
                assert!(
                    (T::tconorm(a, b) - dual).abs() <= TOL,
                    "{}: S({a}, {b}) = {}, 1 - T(1-a, 1-b) = {dual}",
                    T::NAME,
                    T::tconorm(a, b)
                );
            }
        }
    }

    assert_duality::<Godel>();
    assert_duality::<Product>();
    assert_duality::<Lukasiewicz>();
}

#[test]
fn residual_negation_duality_only_recovers_the_lukasiewicz_tconorm() {
    fn by_residual_negation<T: TNorm>(a: f64, b: f64) -> f64 {
        T::negation(T::tnorm(T::negation(a), T::negation(b)))
    }

    for &a in &[0.25, 0.5, 0.75] {
        let derived = by_residual_negation::<Lukasiewicz>(a, 0.5);
        assert!(
            (derived - Lukasiewicz::tconorm(a, 0.5)).abs() <= TOL,
            "lukasiewicz: ¬(¬{a} ⊗ ¬0.5) = {derived}"
        );
    }

    for name in ["godel", "product"] {
        let (derived, tconorm) = match name {
            "godel" => (
                by_residual_negation::<Godel>(0.5, 0.5),
                Godel::tconorm(0.5, 0.5),
            ),
            _ => (
                by_residual_negation::<Product>(0.5, 0.5),
                Product::tconorm(0.5, 0.5),
            ),
        };
        assert!(
            (derived - tconorm).abs() > 1e-3,
            "{name}: the residual negation is not strong, so duality under it \
             cannot recover the t-conorm (got {derived} instead of {tconorm})"
        );
    }
}

#[test]
#[allow(deprecated)]
fn deprecated_catalog_tnorm_matches_the_primary_surface() {
    for (family, lf) in [
        (tnorms::GODEL, LogicFamily::Godel),
        (tnorms::PRODUCT, LogicFamily::Product),
        (tnorms::LUKASIEWICZ, LogicFamily::Lukasiewicz),
    ] {
        for &a in &GRID {
            for &b in &GRID {
                assert_eq!(
                    tnorms::tnorm(family, a, b).to_bits(),
                    lf.tnorm(a, b).to_bits(),
                    "{lf:?}: tnorm({a}, {b}) changed when deprecated"
                );
                assert_eq!(
                    family.tnorm(a, b).to_bits(),
                    lf.tnorm(a, b).to_bits(),
                    "{lf:?}: TConormFamily::tnorm({a}, {b}) changed when deprecated"
                );
            }
        }
    }
}
