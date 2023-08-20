use std::{cell::RefCell, rc::Rc};

use crate::utils::index::increment_dim_index_unchecked;

use super::{storage::TensorStorage, tensor::Tensor};

/// The Iterator struct for Tensors
#[derive(Debug)]
pub struct TensorIterator<'a, T> {
    storage: Rc<RefCell<TensorStorage<T>>>,
    index: Vec<usize>,
    storage_index: usize,
    dims: &'a [usize],
    strides: &'a [usize],
    done: bool,
}

impl<'a, T: Copy> IntoIterator for &'a Tensor<T> {
    type Item = T;

    type IntoIter = TensorIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator {
            storage: Rc::clone(&self.storage),
            index: vec![0; self.no_dim],
            storage_index: self.offset,
            dims: &self.dims,
            strides: &self.strides,
            done: false,
        }
    }
}

impl<'a, T: Copy> Iterator for TensorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let val = self.storage.borrow().get_unchecked(self.storage_index);
        (self.storage_index, self.done) = increment_dim_index_unchecked(
            &mut self.index,
            self.storage_index,
            &self.dims,
            &self.strides,
        );

        Some(val)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::tensor::{storage::TensorStorage, tensor::Tensor};
    use std::{
        cell::RefCell,
        f64::consts::{E, PI, SQRT_2, TAU},
        rc::Rc,
    };

    #[test]
    fn full_tensor_2d() {
        let vector = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 0, &[3, 3], &[3, 1]);
        let tensor_view_vec: Vec<i32> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vector);

        let vector = vec![PI, E, TAU, SQRT_2];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<f64> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 0, &[2, 2], &[2, 1]);
        let tensor_view_vec: Vec<f64> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vector);
    }

    #[test]
    fn with_strides_2d() {
        let vector = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 4, &[2, 2], &[3, 1]);
        let tensor_view_vec: Vec<i32> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![5, 6, 8, 9]);

        let vector = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 1, &[3], &[3]);
        let tensor_view_vec: Vec<i32> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![2, 5, 8]);

        let vector = vec![PI, E, TAU, SQRT_2];
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<f64> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 1, &[2, 1], &[2, 1]);
        let tensor_view_vec: Vec<f64> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![E, SQRT_2]);
    }

    #[test]
    fn with_strides_3d() {
        let vector: Vec<i32> = (0..27).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i32> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 4, &[3], &[9]);
        let tensor_view_vec: Vec<i32> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![4, 13, 22]);

        let vector: Vec<i128> = (0..64).collect();
        let storage = TensorStorage::from_slice(&vector);
        let tensor_view: Tensor<i128> =
            Tensor::new_unchecked(Rc::new(RefCell::new(storage)), 42, &[2, 2, 2], &[16, 4, 1]);
        let tensor_view_vec: Vec<i128> = tensor_view.into_iter().collect();
        assert_eq!(tensor_view_vec, vec![42, 43, 46, 47, 58, 59, 62, 63]);
    }
}
