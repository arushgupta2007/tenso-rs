use num::Complex;
use num_traits::{Float, FloatConst};

use super::log_gamma::{log_gamma, log_gamma_float};

// Calculate gamma(x) by exponentiating log_gamma(x)

pub(crate) fn gamma<R: Copy + Float + FloatConst>(x: Complex<R>) -> Complex<R> {
    return log_gamma(x).exp();
}

pub(crate) fn gamma_float<R: Copy + Float + FloatConst>(x: R) -> R {
    return log_gamma_float(x).exp();
}
