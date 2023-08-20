use num::{complex::ComplexFloat, Complex};
use num_traits::{one, zero, Float, FloatConst, NumCast};

const SMALL: i32 = 7;

// The approximation is based on the implementation of SciPy.
// For small values (< 0.2), this returns the Taylor series expansion of log gamma around 0
// with 23 terms.
// For slightly larger values, this returns the result using a recursive formula and
// the Taylor series expansion like above.
// For big values (> 7), this returns the famous stirling approximation for log gamma
// For values in between, this returns the result by repeatedly applying another recursive
// formula, and finally using the stirling approximation.
// For negative values, this returns the result using a reflection formula
//
// See [1] and [2] for more details
//
// [1] Hare, "Computing the Principal Branch of log-Gamma",
//     Journal of Algorithms, 1997.
// [2] Scipy: https://github.com/scipy/scipy/blob/main/scipy/special/_loggamma.pxd
pub(crate) fn log_gamma<R: Copy + Float + FloatConst>(z: Complex<R>) -> Complex<R> {
    if z.is_nan() || (z.re() <= zero::<R>() && z.re().fract() == zero::<R>()) {
        return Complex::new(R::nan(), R::nan());
    }

    let small: R = NumCast::from(SMALL).unwrap();

    if z.re() > small || z.im().abs() > small {
        return log_gamma_stirling(z);
    }

    let (one, two): (R, R) = (one::<R>(), NumCast::from(2).unwrap());

    let taylor_radius: R = NumCast::from(0.04).unwrap();
    if (z - one).norm_sqr() <= taylor_radius {
        return log_gamma_taylor(z);
    }

    if (z - two).norm_sqr() <= taylor_radius {
        return (z - one).ln() + log_gamma_taylor(z - one);
    }

    if z.re().is_sign_negative() {
        let (pi, tau): (R, R) = (<R as FloatConst>::PI(), <R as FloatConst>::TAU());
        let (half, quater): (R, R) = (NumCast::from(0.5).unwrap(), NumCast::from(0.25).unwrap());
        let sign = if z.im().is_sign_negative() {
            one.neg()
        } else {
            one
        };
        return Complex::new(pi.ln(), sign * tau * (half * z.re() + quater).floor())
            - (Complex::new(pi, zero::<R>()) * z).sin().ln()
            - log_gamma(-z + one);
    }

    if z.im().is_sign_positive() {
        return log_gamma_recurrence(z);
    }

    return log_gamma_recurrence(z.conj()).conj();
}

pub(crate) fn log_gamma_float<R: Copy + Float + FloatConst>(x: R) -> R {
    return log_gamma(Complex::new(x, zero::<R>())).re();
}

fn log_gamma_taylor<R: Copy + Float + FloatConst>(z: Complex<R>) -> Complex<R> {
    let coeff: Vec<Complex<R>> = [
        -4.3478266053040259361e-2,
        4.5454556293204669442e-2,
        -4.7619070330142227991e-2,
        5.000004769810169364e-2,
        -5.2631679379616660734e-2,
        5.5555767627403611102e-2,
        -5.8823978658684582339e-2,
        6.2500955141213040742e-2,
        -6.6668705882420468033e-2,
        7.1432946295361336059e-2,
        -7.6932516411352191473e-2,
        8.3353840546109004025e-2,
        -9.0954017145829042233e-2,
        1.0009945751278180853e-1,
        -1.1133426586956469049e-1,
        1.2550966952474304242e-1,
        -1.4404989676884611812e-1,
        1.6955717699740818995e-1,
        -2.0738555102867398527e-1,
        2.7058080842778454788e-1,
        -4.0068563438653142847e-1,
        8.2246703342411321824e-1,
        -5.7721566490153286061e-1,
    ]
    .iter()
    .map(|&x| NumCast::from(x).unwrap())
    .collect();

    let z = z - one::<R>();

    let mut val = coeff[0] * z + coeff[1];
    for idx in 2..coeff.len() {
        val = val * z + coeff[idx];
    }

    return val * z;
}

fn log_gamma_stirling<R: Copy + Float + FloatConst>(z: Complex<R>) -> Complex<R> {
    let coeff: Vec<Complex<R>> = [
        -2.955065359477124183e-2,
        6.4102564102564102564e-3,
        -1.9175269175269175269e-3,
        8.4175084175084175084e-4,
        -5.952380952380952381e-4,
        7.9365079365079365079e-4,
        -2.7777777777777777778e-3,
        8.3333333333333333333e-2,
    ]
    .iter()
    .map(|&x| NumCast::from(x).unwrap())
    .collect();

    let rz = z.recip();
    let rzz = rz / z;

    let mut val = coeff[0] * rzz + coeff[1];
    for idx in 2..coeff.len() {
        val = val * rzz + coeff[idx];
    }

    let half: Complex<R> = NumCast::from(0.5).unwrap();
    let two: R = NumCast::from(2.0).unwrap();
    let half_log_tau = <R as FloatConst>::TAU().ln() / two;
    let half_log_tau_complex = Complex::new(half_log_tau, zero::<R>());
    return (z - half) * z.ln() - z + half_log_tau_complex + rz * val;
}

fn log_gamma_recurrence<R: Copy + Float + FloatConst>(z: Complex<R>) -> Complex<R> {
    let small: R = NumCast::from(SMALL).unwrap();

    let mut shift_prod = z;
    let mut z = z + one::<Complex<R>>();
    let mut nsb;
    let mut sb = false;
    let mut sign_flips = 0;

    while z.re() <= small {
        shift_prod = shift_prod * z;
        nsb = shift_prod.im().is_sign_negative();
        sign_flips += if nsb && !sb { 1 } else { 0 };
        sb = nsb;
        z = z + one::<Complex<R>>();
    }

    let sign_flips: R = NumCast::from(sign_flips).unwrap();
    let tau = <R as FloatConst>::TAU();

    return log_gamma_stirling(z) - shift_prod.ln() - Complex::new(zero::<R>(), sign_flips * tau);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_integers() {
        let mx = 100000;
        let factorials: Vec<_> = (1..mx)
            .scan(0.0, |state, x| {
                *state = *state + (x as f64).ln();
                Some(*state)
            })
            .collect();
        let res: Vec<_> = (1..mx).map(|x| log_gamma_float((x + 1) as f64)).collect();
        assert!(res
            .iter()
            .zip(factorials.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6));
    }

    #[test]
    fn pos_reals() {
        let x = 8179.208671160341;
        assert_eq!(log_gamma_float(x), 65506.56485017785);

        let x = 6618.112796942345;
        assert_eq!(log_gamma_float(x), 51601.6884026606);

        let x = 3274.018444989861;
        assert_eq!(log_gamma_float(x), 23222.017006272825);

        let x = 2.711883521187877;
        assert_eq!(log_gamma_float(x), 0.444320665580209);

        let x = 4.165513686655052;
        assert_eq!(log_gamma_float(x), 2.0034926482706465);

        let x = 0.8729348890854345;
        assert_eq!(log_gamma_float(x), 0.08752337366548428);

        let x = 1e-5;
        assert_eq!(log_gamma_float(x), 11.512919692895824);
    }

    #[test]
    fn neg_integers() {
        let mx = 10000;
        assert!((1..mx).all(|x| log_gamma_float(-x as f64).is_nan()));
    }

    #[test]
    fn neg_reals() {
        let x = -2886.7646812930543;
        assert_eq!(log_gamma_float(x), -20118.026857374753);

        let x = -227.5960094849852;
        assert_eq!(log_gamma_float(x), -1010.1399232102507);

        let x = -78.40053139355908;
        assert_eq!(log_gamma_float(x), -265.4758597620511);

        let x = -2.7187298106612374;
        assert_eq!(log_gamma_float(x), -0.04784662517865734);

        let x = -0.13110650513541833;
        assert_eq!(log_gamma_float(x), 2.1225512397481163);

        let x = -1e-5;
        assert_eq!(log_gamma_float(x), 11.51293123720912453);
    }

    #[test]
    fn small_complex() {
        let x = Complex::new(0.0019025171497510929, 0.024645715493008397);
        let res = Complex::new(3.6985882637251266, -1.5078973801400546);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(0.03959103464167857, 0.00804462026455055);
        let res = Complex::new(3.1872857624908773, -0.20459786489357734);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(0.0518130173533774, 0.09616780279881729);
        let res = Complex::new(2.1794283590065096, -1.123910390579651);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-0.06876849496680155, 0.04558545860582355);
        let res = Complex::new(2.536715254563112, -2.5879137390616465);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-0.002576019050600231, -0.0371208303473549);
        let res = Complex::new(3.2915302091770076, 1.6616444560602153);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-0.013732427065707764, -0.04078336187134242);
        let res = Complex::new(3.1524693257863965, 1.9200317691715005);
        assert_eq!(log_gamma(x), res);
    }

    #[test]
    fn med_complex() {
        let x = Complex::new(1.3435509407729143, -36.39815386754792);
        let res = Complex::new(-53.22293979505137, -95.7520842488093);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-97.10322834597945, -27.930592139348505);
        let res = Complex::new(-432.3921001087216, 178.3097370063337);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-32.93164155699746, 83.49742532554234);
        let res = Complex::new(-279.0204910335088, 226.92481740386924);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(85.62750183148083, 27.091247976321682);
        let res = Complex::new(289.8695581218488, 120.8417485307509);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-4.592212188760243, 65.18569506497366);
        let res = Complex::new(-122.75102462630014, 198.9137114667023);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(39.41691276500825, -43.871408088513085);
        let res = Complex::new(83.37290618597828, -167.6430960669142);
        assert_eq!(log_gamma(x), res);
    }

    #[test]
    fn big_complex() {
        let x = Complex::new(1144.094412904311, -8403.176055358974);
        let res = Complex::new(-2861.3120154145313, -69249.766113134);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(5795.548242388693, 9614.289802005664);
        let res = Complex::new(38363.385061428264, 86007.59780284636);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-7854.173604381376, 130.74203986415523);
        let res = Complex::new(-63001.55692539907, -23503.571280784075);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-5136.483348349828, -6880.81858702255);
        let res = Complex::new(-56613.86715458049, -44082.06128115435);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(110.19739410719194, 6384.637694227124);
        let res = Complex::new(-9066.910927239787, 49726.69324593493);
        assert_eq!(log_gamma(x), res);

        let x = Complex::new(-8748.145323908942, -7222.278812236434);
        let res = Complex::new(-90637.39117059101, -38761.41109783721);
        assert_eq!(log_gamma(x), res);
    }
}
