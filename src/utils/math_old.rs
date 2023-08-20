use num::{complex::ComplexFloat, Complex};
use num_traits::{one, zero, Float, FloatConst, NumCast};
pub(crate) fn gamma_complex<R: Copy + Float + FloatConst>(z: Complex<R>) -> Complex<R> {
    if z.is_nan()
        || z.is_infinite()
        || (z.im() == zero::<R>() && z.re() <= zero::<R>() && z.re().fract() == zero::<R>())
    {
        return Complex::new(R::nan(), R::nan());
    }

    let pi = <R as FloatConst>::PI();
    if z.re() < NumCast::from(0.5).unwrap() {
        let pi_complex: Complex<R> = pi.into();
        let alt_z = Complex::new(one::<R>() - z.re, -z.im);
        return pi_complex / ((pi_complex * z).sin() * gamma_complex(alt_z));
    }

    let half: Complex<R> = NumCast::from(0.5).unwrap();
    let z = Complex::new(z.re - one::<R>(), z.im);
    let g: Complex<R> = NumCast::from(7).unwrap();

    let p: Vec<Complex<R>> = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    .iter()
    .map(|&x| Complex::new(NumCast::from(x).unwrap(), zero::<R>()))
    .collect();

    let x = p[0]
        + p.iter()
            .enumerate()
            .skip(1)
            .map(|(idx, &x)| {
                let i: Complex<R> = Complex::new(NumCast::from(idx).unwrap(), zero::<R>());
                x / (z + i)
            })
            .sum::<Complex<R>>();

    let t = z + g + half;
    let sqrt_2pi: Complex<R> = <R as FloatConst>::TAU().sqrt().into();
    sqrt_2pi * t.powc(z + half) * (-t).exp() * x
}

pub fn gamma<R: Copy + Float + FloatConst>(x: R) -> R {
    if x.is_nan() || x.is_infinite() || (x <= zero::<R>() && x.fract() == zero::<R>()) {
        R::nan()
    } else {
        gamma_complex(x.into()).re
    }
}
