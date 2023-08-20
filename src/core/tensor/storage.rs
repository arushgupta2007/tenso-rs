use crate::utils::errors::Errors;

/// The Storage of data held in each Tensor
#[derive(Debug)]
pub struct TensorStorage<T> {
    arr: Vec<T>,
    sz: usize,
}

impl<T: Copy> TensorStorage<T> {
    /// Create new Tensor Storage with `len` elements each equal to `val`
    pub fn from_val(len: usize, value: T) -> TensorStorage<T> {
        TensorStorage {
            arr: vec![value; len],
            sz: len,
        }
    }

    /// Create new Tensor Storage from `vec`
    pub fn from_slice(vec: &[T]) -> TensorStorage<T> {
        TensorStorage {
            arr: vec.to_vec(),
            sz: vec.len(),
        }
    }

    /// Get number of elements in `self`
    pub fn len(&self) -> usize {
        self.sz
    }

    /// Get element in `self` on index `idx`
    pub fn get(&self, idx: usize) -> Result<T, Errors> {
        if idx < self.sz {
            Ok(self.arr[idx])
        } else {
            Err(Errors::OutOfBounds {
                expected: self.arr.len(),
                found: idx,
                axis: 0,
            })
        }
    }

    /// Get element in `self` on index `idx` without any checks
    pub fn get_unchecked(&self, idx: usize) -> T {
        self.arr[idx]
    }

    /// Update element in `self` on index `idx` to `val`
    pub fn upd(&mut self, idx: usize, new_val: T) -> Result<(), Errors> {
        if idx < self.sz {
            self.arr[idx] = new_val;
            Ok(())
        } else {
            Err(Errors::OutOfBounds {
                expected: self.arr.len(),
                found: idx,
                axis: 0,
            })
        }
    }

    /// Update element in `self` on index `idx` to `val` without any checks
    pub fn upd_unchecked(&mut self, idx: usize, new_val: T) {
        self.arr[idx] = new_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::errors::Errors;

    #[test]
    fn create_new() {
        let tensor_storage = TensorStorage::from_val(10000000, 3723_i32);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert_eq!(*x, 3723_i32));

        let tensor_storage = TensorStorage::from_val(10000000, 1_000_000_000_000_000_000_i64);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert_eq!(*x, 1_000_000_000_000_000_000_i64));

        let pi = std::f64::consts::PI;
        let eps = 0.000_000_000_1;
        let tensor_storage = TensorStorage::from_val(10000000, pi);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert!((*x - pi).abs() < eps));
    }

    #[test]
    fn create_new_from_vec() {
        let vec = vec![3723_i32; 10000000];
        let tensor_storage = TensorStorage::from_slice(&vec);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert_eq!(*x, 3723_i32));

        let vec = vec![1_000_000_000_000_000_000_i64; 10000000];
        let tensor_storage = TensorStorage::from_slice(&vec);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert_eq!(*x, 1_000_000_000_000_000_000_i64));

        let pi = std::f64::consts::PI;
        let eps = 0.000_000_000_1;
        let vec = vec![pi; 10000000];
        let tensor_storage = TensorStorage::from_slice(&vec);
        tensor_storage
            .arr
            .iter()
            .for_each(|x| assert!((*x - pi).abs() < eps));

        let vec = vec![1, 4, 9, 16, 25, 36, 49];
        let tensor_storage = TensorStorage::from_slice(&vec);
        assert_eq!(vec, tensor_storage.arr);

        let mut vec = vec![0_i64, 1_i64];
        for _ in 2..88 {
            vec.push(vec[vec.len() - 1] + vec[vec.len() - 2]);
        }
        let tensor_storage = TensorStorage::from_slice(&vec);
        assert_eq!(vec, tensor_storage.arr);
    }

    #[test]
    fn get() {
        let tensor_storage = TensorStorage::from_val(10000000, 3723_i32);
        (0..10000000).for_each(|x| assert_eq!(tensor_storage.get_unchecked(x), 3723_i32));
        (0..10000000).for_each(|x| assert_eq!(tensor_storage.get(x).unwrap(), 3723_i32));
        assert!(match tensor_storage.get(10000001) {
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

        let tensor_storage = TensorStorage::from_val(10000000, 1_000_000_000_000_000_000_i64);
        (0..10000000).for_each(|x| {
            assert_eq!(
                tensor_storage.get_unchecked(x),
                1_000_000_000_000_000_000_i64
            )
        });
        (0..10000000).for_each(|x| {
            assert_eq!(
                tensor_storage.get(x).unwrap(),
                1_000_000_000_000_000_000_i64
            )
        });
        assert!(match tensor_storage.get(10000001) {
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

        let pi = std::f64::consts::PI;
        let eps = 0.000_000_000_1;
        let tensor_storage = TensorStorage::from_val(10000000, pi);
        (0..10000000).for_each(|x| assert!((tensor_storage.get_unchecked(x) - pi).abs() < eps));
        (0..10000000).for_each(|x| assert!((tensor_storage.get(x).unwrap() - pi).abs() < eps));
        assert!(match tensor_storage.get(10000001) {
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
        let mut vec = vec![1, 4, 9, 16, 25, 36, 49];
        let mut tensor_storage = TensorStorage::from_slice(&vec);
        tensor_storage.upd_unchecked(1, 5);
        vec[1] = 5;
        assert_eq!(vec, tensor_storage.arr);
        assert!(match tensor_storage.upd(10, 10) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: e,
                    found: f,
                    axis: a,
                } => e == 7 && f == 10 && a == 0,
                _ => false,
            },
        });

        let mut vec = vec![0_i64, 1_i64];
        for _ in 2..88 {
            vec.push(vec[vec.len() - 1] + vec[vec.len() - 2]);
        }
        let mut tensor_storage = TensorStorage::from_slice(&vec);
        tensor_storage.upd_unchecked(3, 0);
        vec[3] = 0_i64;
        assert_eq!(vec, tensor_storage.arr);
        assert!(match tensor_storage.upd(100, 100) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::OutOfBounds {
                    expected: e,
                    found: f,
                    axis: a,
                } => e == 88 && f == 100 && a == 0,
                _ => false,
            },
        });
    }
}
