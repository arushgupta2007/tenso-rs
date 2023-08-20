use num_traits::{one, Float, NumCast};

// Current approximation method (due to Abramowitz and Stegun):
// erf(x) ~ 1 - (a1 * t + a2 * t^2 + a3 * t^3 + a4 * t^4 + a5 * t^5) * e^-(x^2)
// where t = 1 / (1 + p * x), p = 0.3275911, a1 = 0.254829592, a2 = −0.284496736, a3 = 1.421413741,
// a4 = −1.453152027, a5 = 1.061405429
//
// More information can be found on wikipedia page for the error function
pub(crate) fn erf<T: Float>(x: T) -> T {
    let sign = if x.is_nan() || x.is_sign_negative() {
        one::<T>().neg()
    } else {
        one::<T>()
    };
    let a: Vec<T> = [
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    ]
    .iter()
    .map(|&y| NumCast::from(y).unwrap())
    .collect();
    let p: T = NumCast::from(0.3275911).unwrap();

    let x = x.abs();
    let t = (one::<T>() + p * x).recip();
    sign * (one::<T>()
        - ((((a[4] * t + a[3]) * t + a[2]) * t + a[1]) * t + a[0]) * t * (-x * x).exp())
}

// erfc(x) = 1 - erf(x)
pub(crate) fn erfc<T: Float>(x: T) -> T {
    return one::<T>() - erf(x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        assert!((erf(0.08815347519813702) - 0.0992134814684843).abs() < 1e-6);
        assert!((erf(-0.01752647555037734) - -0.0197744851049801).abs() < 1e-6);
        assert!((erf(-0.24280421430057686) - -0.2686851494828099).abs() < 1e-6);
        assert!((erf(-0.6599570591087505) - -0.6493453435785218).abs() < 1e-6);
        assert!((erf(0.6612412310378295) - 0.6502819473899006).abs() < 1e-6)
    }
}
