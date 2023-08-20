use num_traits::{Float, NumCast};
use std::{cell::RefCell, fmt::Display, ops, rc::Rc};

use crate::utils::{errors::Errors, strides::new_strides_from_dim};

use super::{storage::TensorStorage, tensor::Tensor};

impl<T: Copy> Tensor<T> {
    /// Create a one dimensional tensor from a specified slice
    ///
    /// # Arguments
    /// * arr - The slice containing the data
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).unwrap();
    /// // t = [1, 2, 3, 4, 5, 6]
    /// ```
    pub fn from_slice(arr: &[T]) -> Result<Tensor<T>, Errors> {
        if arr.is_empty() {
            return Err(Errors::EmptyTensor);
        }
        let tensor_storage = TensorStorage::<T>::from_slice(&arr);
        let dims = vec![arr.len()];
        let strides = vec![1];
        Ok(Tensor::new_unchecked(
            Rc::new(RefCell::new(tensor_storage)),
            0,
            &dims,
            &strides,
        ))
    }

    /// Create a new tensor with specified data in a slice, with specified dimensions
    ///
    /// # Arguments
    /// * arr - The slice containing the data
    /// * dims - The required dimensions
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[2, 5]).unwrap();
    /// // t = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    /// ```
    pub fn from_slice_and_dims(arr: &[T], dims: &[usize]) -> Result<Tensor<T>, Errors> {
        if arr.is_empty() {
            return Err(Errors::EmptyTensor);
        }

        if arr.len() != dims.iter().product() {
            return Err(Errors::InputError(
                "Expected product of dimensions to equal number of elements in storage".to_string(),
            ));
        }
        let tensor_storage = TensorStorage::<T>::from_slice(&arr);
        let strides = new_strides_from_dim(&dims);
        Ok(Tensor::new_unchecked(
            Rc::new(RefCell::new(tensor_storage)),
            0,
            &dims,
            &strides,
        ))
    }

    /// Return a new tensor with all elements equal to a specified value, with specified dimensions
    ///
    /// # Arguments
    /// * dims - The required dimensions
    /// * val - The value
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::from_val(&[2, 2], 1729).unwrap();
    /// // t = [[1729, 1729], [1729, 1729]]
    /// ```
    pub fn from_val(dims: &[usize], val: T) -> Result<Tensor<T>, Errors> {
        let no_el = dims.iter().fold(1, |res, dim_sz| res * dim_sz);
        if no_el == 0 {
            return Err(Errors::EmptyTensor);
        }

        let tensor_storage = TensorStorage::<T>::from_val(no_el, val);
        let new_strides = new_strides_from_dim(&dims);
        Ok(Tensor::new_unchecked(
            Rc::new(RefCell::new(tensor_storage)),
            0,
            &dims,
            &new_strides,
        ))
    }
}

impl<T: Default + Copy> Tensor<T> {
    /// Return a tensor with all elements equal to the type's default value, with specified
    /// dimensions
    ///
    /// # Arguments
    /// * dims - The required dimensions
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::<i32>::from_default(&[2, 3]).unwrap();
    /// // t = [[0, 0, 0], [0, 0, 0]]
    /// ```
    pub fn from_default(dims: &[usize]) -> Result<Tensor<T>, Errors> {
        Tensor::from_val(dims, T::default())
    }
}

impl<T: Copy + NumCast> Tensor<T> {
    /// Return a tensor with all elements equal to 0, with specified dimensions
    ///
    /// # Arguments
    /// * dims - The required dimensions
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::<i32>::zeros(&[2, 3]).unwrap();
    /// // t = [[0, 0, 0], [0, 0, 0]]
    pub fn zeros(dims: &[usize]) -> Result<Tensor<T>, Errors> {
        Tensor::<T>::from_val(&dims, NumCast::from(0).unwrap())
    }

    /// Return a tensor with all elements equal to 1, with specified dimensions
    ///
    /// # Arguments
    /// * dims - The required dimensions
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::<i32>::ones(&[2, 3]).unwrap();
    /// // t = [[1, 1, 1], [1, 1, 1]]
    pub fn ones(dims: &[usize]) -> Result<Tensor<T>, Errors> {
        Tensor::<T>::from_val(&dims, NumCast::from(1).unwrap())
    }

    /// Return a 2D tensor with elements equal to 0 except the main diagonal (when row = column)
    /// which is equal to 1
    ///
    /// # Arguments
    /// * n - Number of rows
    /// * m - Number of columns
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::<i32>::eye(2, 3).unwrap();
    /// // t = [[1, 0, 0], [0, 1, 0]]
    /// ```
    pub fn eye(n: usize, m: usize) -> Result<Tensor<T>, Errors> {
        if n == 0 || m == 0 {
            return Err(Errors::EmptyTensor);
        }

        let tensor = Tensor::<T>::zeros(&[n, m])?;
        (0..(n.min(m))).for_each(|i| {
            tensor.upd_unchecked(&[i, i], NumCast::from(1).unwrap());
        });
        Ok(tensor)
    }
}

impl<T: Copy + ops::Add<Output = T> + ops::Mul<Output = T> + NumCast + PartialOrd> Tensor<T> {
    /// Return a 1D tensor where the i th element is `st` + i * `step` such that each element is in
    /// the range from `st` to `en` (exclusive)
    ///
    /// # Arguments
    /// * st - The start of the range
    /// * en - The end of the range
    /// * step - The step size
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 11, 1).unwrap();
    /// // t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    /// ```
    pub fn arange(st: T, en: T, step: T) -> Result<Tensor<T>, Errors> {
        let res: Vec<T> = (0..)
            .map(|i| <T as NumCast>::from(i).unwrap() * step + st)
            .take_while(|&v| v < en)
            .collect();
        Tensor::from_slice(&res)
    }
}

impl<T: Copy + Float + Display> Tensor<T> {
    /// Returns a 1D tensor with `cnt` elements equally spaced in the range from `st` to `en`
    ///
    /// # Arguments
    /// * st - Start of the range
    /// * en - End of the range
    /// * cnt - The number of elements in the resultant tensor
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::linspace(0.0, 1.0, 11).unwrap();
    /// // t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    /// ```
    pub fn linspace(st: T, en: T, cnt: usize) -> Result<Tensor<T>, Errors> {
        if cnt == 0 {
            return Err(Errors::EmptyTensor);
        }

        if en <= st {
            return Err(Errors::InputError(format!(
                "linspace expected st < en, found {} >= {}",
                st, en
            )));
        }

        if cnt == 1 {
            return Tensor::from_slice(&[st]);
        }

        let step_sz = (en - st) / NumCast::from(cnt - 1).unwrap();
        let res: Vec<T> = (0..cnt)
            .map(|x| st + <T as NumCast>::from(x).unwrap() * step_sz)
            .collect();
        Tensor::from_slice(&res)
    }

    /// Return a 1D Tensor of size `cnt` with elements evenly spaced from `base`^`st` to
    /// `base`^`en` on a logarithmic scale with base `base`
    ///
    /// # Arguments
    /// * base - Base for the logarithmic scale
    /// * st - Start of the range
    /// * en - End of the range
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::logspace(10.0, -10.0, 10.0, 5).unwrap();
    /// // t = [1e-10, 1e-5, 1, 1e5, 1e10]
    /// ```
    pub fn logspace(base: T, st: T, en: T, cnt: usize) -> Result<Tensor<T>, Errors> {
        if cnt == 0 {
            return Err(Errors::EmptyTensor);
        }

        if en <= st {
            return Err(Errors::InputError(format!(
                "logspace expected st < en, found {} >= {}",
                st, en
            )));
        }

        if cnt == 1 {
            return Tensor::from_slice(&[base.powf(st)]);
        }

        let step_sz = (en - st) / NumCast::from(cnt - 1).unwrap();
        let res: Vec<T> = (0..cnt)
            .map(|x| st + <T as NumCast>::from(x).unwrap() * step_sz)
            .map(|x| base.powf(x))
            .collect();
        Tensor::from_slice(&res)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use crate::{core::tensor::tensor::Tensor, utils::errors::Errors};

    #[test]
    fn arange() {
        let t = Tensor::<u128>::arange(12, 37, 7).unwrap();
        let exp = vec![12, 19, 26, 33];
        assert_eq!(t.into_iter().collect::<Vec<u128>>(), exp);

        let t = Tensor::<f64>::arange(-10.0, -9.0, 0.1).unwrap();
        let exp = vec![-10.0, -9.9, -9.8, -9.7, -9.6, -9.5, -9.4, -9.3, -9.2, -9.1];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        assert!(match Tensor::arange(1, -1, 1) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });
    }

    #[test]
    fn linspace() {
        let t = Tensor::<f32>::linspace(-10.0, 10.0, 5).unwrap();
        let exp = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        let t = Tensor::<f32>::linspace(1.0, 10.0, 21).unwrap();
        let exp = vec![
            1., 1.45, 1.9, 2.35, 2.8, 3.25, 3.7, 4.15, 4.6, 5.05, 5.5, 5.95, 6.4, 6.85, 7.3, 7.75,
            8.2, 8.65, 9.1, 9.55, 10.,
        ];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        assert!(match Tensor::linspace(1.0, 2.0, 0) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });

        assert!(match Tensor::linspace(1.0, -2.0, 2) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        })
    }

    #[test]
    fn logspace() {
        let t = Tensor::<f32>::logspace(10.0, -10.0, 10.0, 5).unwrap();
        let exp = vec![1e-10, 1e-5, 1.0, 1e5, 1e10];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        let t = Tensor::<f64>::logspace(E, 1.0, 5.0, 5).unwrap();
        let exp = vec![E, E.powi(2), E.powi(3), E.powi(4), E.powi(5)];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        assert!(match Tensor::logspace(E, 1.0, 2.0, 0) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });

        assert!(match Tensor::logspace(E, 1.0, -2.0, 2) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        })
    }

    #[test]
    fn eye() {
        let t = Tensor::<i128>::eye(2, 5).unwrap();
        let exp = vec![1, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(t.into_iter().collect::<Vec<i128>>(), exp);

        let t = Tensor::<f64>::eye(4, 3).unwrap();
        let exp = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert!(t
            .into_iter()
            .zip(exp.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001));

        assert!(match Tensor::<f32>::eye(0, 10) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });
    }
}
