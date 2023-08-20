use num_traits::{one, Float, FloatConst};

// By Definition of Normalized Sinc Function:
// sinc(x) = sin(pi * x) / (pi * x) for x != 0
// sinc(0) = 1
pub(crate) fn sinc<T: Float + FloatConst>(x: T) -> T {
    if x.is_zero() {
        one::<T>()
    } else {
        let inner: T = <T as FloatConst>::PI() * x;
        inner.sin() / inner
    }
}
