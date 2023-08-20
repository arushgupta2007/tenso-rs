use num_traits::{one, Float};

pub(crate) fn logit<T: Float>(x: T) -> T {
    (x / (one::<T>() - x)).ln()
}

