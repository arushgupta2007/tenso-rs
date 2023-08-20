use crate::utils::{
    errors::Errors,
    index::{dim_index_to_storage_index, dim_index_to_storage_index_unchecked},
};
use std::{cell::RefCell, ops::Range, rc::Rc};

use super::storage::TensorStorage;

/// The N Dimensional Array
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub(crate) storage: Rc<RefCell<TensorStorage<T>>>,
    pub(crate) no_dim: usize,
    pub(crate) no_el: usize,
    pub(crate) offset: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: Copy> Tensor<T> {
    /// Create new tensor with custom TensorStorage, offset, dimensions, and strides.
    /// Not recommended for use
    pub fn new(
        storage: Rc<RefCell<TensorStorage<T>>>,
        offset: usize,
        dims: &[usize],
        strides: &[usize],
    ) -> Result<Tensor<T>, Errors> {
        // TODO: Check if any element is outside storage limit
        if dims.len() != strides.len() {
            Err(Errors::DimsNeqStrides {
                dim_len: dims.len(),
                strides_len: strides.len(),
            })
        } else if dims.is_empty() || dims.iter().any(|&x| x == 0) {
            Err(Errors::EmptyTensor)
        } else {
            let last_idx: Vec<_> = dims.iter().map(|&x| x - 1).collect();
            let last_storage_idx = dim_index_to_storage_index(&last_idx, offset, &dims, &strides)?;
            storage.borrow().get(last_storage_idx)?;
            Ok(Tensor::new_unchecked(storage, offset, dims, strides))
        }
    }

    /// Create new tensor with custom TensorStorage, offset, dimensions, and strides, without any
    /// checks
    /// Not recommended for use
    pub fn new_unchecked(
        storage: Rc<RefCell<TensorStorage<T>>>,
        offset: usize,
        dims: &[usize],
        strides: &[usize],
    ) -> Tensor<T> {
        let no_dim = dims.len();
        let no_el = dims.iter().fold(1, |res, dim_sz| res * dim_sz);
        Tensor {
            storage,
            no_dim,
            no_el,
            offset,
            dims: dims.to_vec(),
            strides: strides.to_vec(),
        }
    }

    /// Get pointer to `self.storage`
    pub fn get_storage_ptr(&self) -> Rc<RefCell<TensorStorage<T>>> {
        Rc::clone(&self.storage)
    }

    /// Get number of dimensions
    pub fn no_dim(&self) -> usize {
        self.no_dim
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.no_el
    }

    /// Does `self` borrow memory from other tensors
    pub fn is_view(&self) -> bool {
        self.no_el != self.storage.borrow().len()
    }

    /// Make `self` own it's own data
    pub fn make_contiguous(&self) -> Tensor<T> {
        let res: Vec<_> = self.into_iter().collect();
        Tensor::from_slice_and_dims(&res, &self.dims).unwrap()
    }

    /// Index `self` with `rngs`
    pub fn slice(&self, rngs: &[Range<usize>]) -> Result<Tensor<T>, Errors> {
        if rngs.len() != self.no_dim {
            Err(Errors::InvalidIndexSize {
                expected: self.no_dim,
                found: rngs.len(),
            })
        } else if let Some(idx) = rngs
            .iter()
            .zip(self.dims.iter())
            .position(|(rng, &dim)| rng.end > dim)
        {
            Err(Errors::OutOfBounds {
                expected: self.dims[idx],
                found: rngs[idx].end,
                axis: idx,
            })
        } else if rngs.iter().any(|rng| rng.is_empty()) {
            Err(Errors::EmptyTensor)
        } else {
            Ok(self.slice_unchecked(rngs))
        }
    }

    /// Index `self` with `rngs` without checks
    pub fn slice_unchecked(&self, rngs: &[Range<usize>]) -> Tensor<T> {
        let new_offset = self.offset
            + self
                .strides
                .iter()
                .zip(rngs.iter())
                .fold(0, |res, (&stride, rng)| res + stride * rng.start);
        let new_dims: Vec<usize> = rngs
            .iter()
            .map(|rng| (rng.end - rng.start).max(1))
            .collect();
        Tensor::new_unchecked(
            Rc::clone(&self.storage),
            new_offset,
            &new_dims,
            &self.strides,
        )
    }

    /// Get value in `self` at index `index`
    pub fn at(&self, index: &[usize]) -> Result<T, Errors> {
        let storage_idx =
            dim_index_to_storage_index(&index, self.offset, &self.dims, &self.strides)?;
        self.storage.borrow().get(storage_idx)
    }

    /// Get value in `self` at index `index` without checks
    pub fn at_unchecked(&self, index: &[usize]) -> T {
        let storage_idx = dim_index_to_storage_index_unchecked(&index, self.offset, &self.strides);
        self.storage.borrow().get_unchecked(storage_idx)
    }

    /// Update value in `self` at index `index` to `new_val`
    pub fn upd(&self, index: &[usize], new_val: T) -> Result<(), Errors> {
        let storage_idx =
            dim_index_to_storage_index(&index, self.offset, &self.dims, &self.strides)?;
        self.storage.borrow_mut().upd(storage_idx, new_val)?;
        Ok(())
    }

    /// Update value in `self` at index `index` to `new_val` without checks
    pub fn upd_unchecked(&self, index: &[usize], new_val: T) {
        let storage_idx = dim_index_to_storage_index_unchecked(&index, self.offset, &self.strides);
        self.storage
            .borrow_mut()
            .upd_unchecked(storage_idx, new_val);
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI, SQRT_2, TAU};

    use super::*;

    #[test]
    fn splice_1d() {
        let vector: Vec<i32> = (0..27).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 0, &[vector.len()], &[1]);
        let sliced = tensor_view.slice(&[4..7]).unwrap();
        let tensor_view_vec: Vec<i32> = sliced.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![4, 5, 6]);
        assert!(match tensor_view.slice(&[4..4]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });

        let vector = vec![PI, E, TAU, SQRT_2];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<f64> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 0, &[vector.len()], &[1]);
        let sliced = tensor_view.slice(&[1..2]).unwrap();
        let tensor_view_vec: Vec<f64> = sliced.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![E]);
        assert_eq!(sliced.dims, vec![1]);
        assert!(match tensor_view.slice(&[4..5]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });
    }

    #[test]
    fn splice_3d() {
        let vector: Vec<i32> = (0..27).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 4, &[3, 2, 2], &[9, 3, 1]);
        let sliced = tensor_view.slice(&[1..2, 0..2, 1..2]).unwrap();
        let tensor_view_vec: Vec<i32> = sliced.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![14, 17]);
        assert_eq!(sliced.dims, vec![1, 2, 1]);
        assert!(match tensor_view.slice(&[4..5, 4..5, 4..5]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });

        let vector: Vec<i128> = (0..64).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i128> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 42, &[2, 2, 2], &[16, 4, 1]);
        let sliced = tensor_view.slice(&[1..2, 1..2, 1..2]).unwrap();
        let tensor_view_vec: Vec<i128> = sliced.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![63]);
        assert_eq!(sliced.dims, vec![1, 1, 1]);
        assert!(match tensor_view.slice(&[0..0, 0..0, 0..0]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });
    }

    #[test]
    fn get() {
        let vector: Vec<i32> = (0..27).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 4, &[3, 2, 2], &[9, 3, 1]);
        assert_eq!(tensor_view.at(&[2, 1, 1]).unwrap(), 26);
        assert!(match tensor_view.at(&[2, 1, 2]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });

        let vector: Vec<i128> = (0..64).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i128> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 42, &[2, 2, 2], &[16, 4, 1]);
        assert_eq!(tensor_view.at(&[1, 1, 0]).unwrap(), 62);
        assert!(match tensor_view.at(&[2, 1, 0]) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });
    }

    #[test]
    fn upd() {
        let vector: Vec<i32> = (0..27).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 4, &[3, 2, 2], &[9, 3, 1]);
        tensor_view.upd(&[2, 1, 1], -100).unwrap();
        assert_eq!(tensor_view.at(&[2, 1, 1]).unwrap(), -100);
        assert!(match tensor_view.upd(&[2, 1, 2], -100) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });

        let vector: Vec<i128> = (0..64).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i128> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 42, &[2, 2, 2], &[16, 4, 1]);
        tensor_view.upd(&[1, 1, 0], -100).unwrap();
        assert_eq!(tensor_view.at(&[1, 1, 0]).unwrap(), -100);
        assert!(match tensor_view.upd(&[2, 1, 0], -100) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: _,
                    found: _,
                    axis: _,
                } => true,
                _ => false,
            },
        });
    }
}
