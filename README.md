# Tenso-rs

The `tenso-rs` crate provides functionality to work with N dimensional tensors, similar to NumPy, PyTorch, TensorFlow, the `ndarray` crate, etc.

This is a toy project, for me to explore how these amazing libraries work under the hood. *This is by no means ready for use aside from exploration or being a laughing stock.*

## Highlights
- N dimensional Tensors with support for custom types (0 != N for now, 0 dimensional tensor coming soon)
- Common tensor creation methods like `zeros`, `linespace`, `arange`, etc.
- Tensor Views
- Basic Tensor manipulation methods like `cat`, `reshape`, `permute`, `transpose`, etc.
- Views of Tensors; Tensor Slicing; Tensor Iterators
- Common math operations like `cos`, `arctanh`, `sqrt`, `clamp`
	- Few "Advanced" functions like `erfc`, `sinc`, `log_gamma`


## TODOs

Stuff I want to do soon:
- "Good" Macro for tensor creation with user data
- Zero Dimensional Tensors
- Inplace Tensor operations
- Broadcasting
- More Tensor manipulation methods (from [numpy API docs](https://numpy.org/doc/stable/reference/arrays.ndarray.html))
- More math operations, (from [scipy special functions](https://docs.scipy.org/doc/scipy/reference/special.html))
- Save and Load using `serde`
- Optimize this slow code (source code not even bench-marked)
- Linear Algebra and Tensor multiplication
- Integration with BLAS and `matrixmultiply` crate / custom code
- And more.

## Examples
You can create new tensors from the (few) creations methods like so:
```rust
// The following represents the tensor: [1, 2, 3, 4, 5, 6, 7, 8, 9]
let t1 = Tensor::<u128>::arange(1, 10, 1).unwrap();

// The following represents the tensor: [-10.0, -5.0, 0.0, 5.0, 10.0]
let t2 = Tensor::<f64>::linespace(-10.0, 10.0, 5).unwrap();

// The following represents the tensor: [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
let t3 = Tensor::<i8>::eye(2, 3).unwrap();

// The following represents the tensor: [[1, 2, 3], [4, 5, 6]].
// As the TODOS mention, a nice macro for tensor from user data does not exist right now.
let t4 = Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();

// There are (a few) more!
```

Modification of tensors:
```rust
let t1 = Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[3, 4]).unwrap();
let t2 = Tensor::from_slice_and_dims(&[13, 14, 15, 16, 17, 18], &[3, 2]).unwrap();
let res = t1.cat(&t2, 1).unwrap();
// res represents [[1, 2, 3, 4, 13, 14], [5, 6, 7, 8, 15, 16], [9, 10, 11, 12, 17, 18]]

let t = Tensor::<f64>::arange(0, 9, 1).unwrap();
let res = t.reshape(&[3, 3]).unwrap();
// res represents [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

let t = Tensor::arange(0, 24, 1).unwrap().reshape(&[2, 3, 4]).unwrap();
let res = t.permute(&[2, 0, 1]).unwrap();
// res represents [[[0, 4, 8], [12, 16, 20]], [[1, 5, 9], [13, 17, 21]], [[2, 6, 10], [14, 18, 22]], [[3, 7, 11], [15, 19, 23]]]

// There are (a few) more!
```

Math Operations:
```rust
let t = Tensor::logspace(f64::consts::E, 0.0, 5.0, 6).unwrap();
let res = t.cos();
// res represents [cos(0), cos(e), cos(e^2), cos(e^3), cos(e^4), cos(e^5)]

let t = Tensor::from_slice_and_dims(&[1.0, 4.0, 9.0, 16.0, 25.0], &[5]).unwrap();
let res = t.rsqrt();
// res represents [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5]


let t = Tensor::arange(1.0, 6.0, 1.0).unwrap();
let res = t.gammaf();
// res represents [1.0, 1.0, 2.0, 6.0, 24.0, 120.0]

// There are (a few) more!
```

## Notes
- Untested: Arbitrary Precision is possible using `num_bigint`
	- Note: "Advanced" Math operations like `erf` and `gamma` are not arbitrary precision yet.
- New issues, pull requests, or ideas are more than welcome!
