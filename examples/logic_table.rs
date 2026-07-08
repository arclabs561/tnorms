use tnorms::LogicFamily;

fn main() {
    let inputs = [(0.2, 0.8), (0.4, 0.7), (0.8, 0.3)];
    let families = [
        ("godel", LogicFamily::Godel),
        ("product", LogicFamily::Product),
        ("lukasiewicz", LogicFamily::Lukasiewicz),
    ];

    println!("family       a     b     and    or     a->b   not(a)");
    for (name, family) in families {
        for (a, b) in inputs {
            println!(
                "{name:<11} {a:.2}  {b:.2}  {:.3}  {:.3}  {:.3}  {:.3}",
                family.tnorm(a, b),
                family.tconorm(a, b),
                family.residuum(a, b),
                family.neg(a),
            );
        }
    }

    let product = LogicFamily::Product.tnorm(0.4, 0.7);
    assert!((product - 0.28).abs() < 1e-12);
    assert_eq!(LogicFamily::Godel.residuum(0.8, 0.3), 0.3);
    assert_eq!(LogicFamily::Lukasiewicz.neg(0.8), 0.19999999999999996);
}
