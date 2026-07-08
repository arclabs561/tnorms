use tnorms::{
    log_product, owa, power_mean, product_from_log, representable_uninorm, LogicFamily,
    MixedRegion, OrdinalSegment, OrdinalSum, TConormFamily,
};

fn main() {
    let evidence = [0.98, 0.91, 0.87, 0.72, 0.64];

    let direct = evidence
        .iter()
        .copied()
        .fold(1.0, |acc, degree| LogicFamily::Product.tnorm(acc, degree));
    let log_degree = log_product(&evidence, 1e-12);
    let from_log = product_from_log(log_degree);
    assert!((direct - from_log).abs() < 1e-12);

    let mut ranked = evidence;
    let optimistic = owa(&[0.55, 0.25, 0.10, 0.07, 0.03], &mut ranked).unwrap();
    let arithmetic = power_mean(&evidence, 1.0).unwrap();
    let bottleneck = power_mean(&evidence, f64::NEG_INFINITY).unwrap();

    let uninorm = representable_uninorm(
        TConormFamily::Probabilistic,
        TConormFamily::Probabilistic,
        MixedRegion::Min,
        0.5,
        0.35,
        0.82,
    );

    let segments = [
        OrdinalSegment::new(0.0, 0.4, TConormFamily::Bounded),
        OrdinalSegment::new(0.6, 1.0, TConormFamily::Probabilistic),
    ];
    let ordinal = OrdinalSum::new(&segments).unwrap();
    let ordinal_and = ordinal.tnorm(0.72, 0.88);

    println!("product direct:     {direct:.6}");
    println!("product log-space:  {from_log:.6}");
    println!("owa optimistic:     {optimistic:.6}");
    println!("arithmetic mean:    {arithmetic:.6}");
    println!("bottleneck mean:    {bottleneck:.6}");
    println!("mixed uninorm:      {uninorm:.6}");
    println!("ordinal-sum and:    {ordinal_and:.6}");

    assert!(optimistic >= arithmetic);
    assert_eq!(bottleneck, 0.64);
    assert!((0.0..=1.0).contains(&uninorm));
    assert!((0.0..=1.0).contains(&ordinal_and));
}
