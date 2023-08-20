use thiserror::Error;

/// All possible Errors that could be encountered
#[derive(Error, Debug)]
pub enum Errors {
    /// When an index is out of bounds
    #[error("Index Out Of Bounds, expected < {expected}, found {found} on axis {axis}")]
    OutOfBounds {
        /// Expected maximum index
        expected: usize,
        /// What you entered
        found: usize,
        /// The axis in which the error occurred
        axis: usize,
    },

    /// When the size of the dimensions is incorrect
    #[error("Incorrect Size of Dimensional Index, expected {expected}, found {found}")]
    InvalidIndexSize {
        /// Expected size of dimensions
        expected: usize,
        /// What you entered
        found: usize
    },

    /// When the strides and dimensions do not match
    #[error("Number of Dimensions does not match Number of Strides")]
    DimsNeqStrides {
        /// The length of the dimensions
        dim_len: usize,
        /// The length of the strides
        strides_len: usize
    },

    /// When you input something wrong
    #[error("Input Error: {0}")]
    InputError(String),

    /// When an operation returns an empty tensor
    ///
    /// This is because `tenso-rs` does not support zero dimensional or empty tensors yet
    /// This may be gone in the future
    #[error("Empty Tensor")]
    EmptyTensor,
}
