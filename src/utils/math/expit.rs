use num_traits::{one, Float};

pub(crate) fn expit<T: Float>(x: T) -> T {
    (one::<T>() + (-x).exp()).recip()
}
