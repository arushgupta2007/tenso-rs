use num_traits::Float;

// Exactly what is sounds like:
// xlog(y) = x * log(y)
pub(crate) fn xlogy<T: Float>(x: T, y: T) -> T {
    x * y.ln()
}
