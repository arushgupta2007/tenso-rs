use std::{cell::RefCell, rc::Rc};

use num_traits::NumCast;

use crate::utils::{
    errors::Errors, index::increment_dim_index_unchecked, strides::new_strides_from_dim,
};

use super::{storage::TensorStorage, tensor::Tensor};

impl<T: Copy + PartialEq + NumCast> Tensor<T> {
    /// Returns a new tensor with indices for non zero elements in `self`
    ///
    /// # Example
    /// ```rust
    /// let indices = t.argwhere();
    /// ```
    pub fn argwhere(&self) -> Result<Tensor<usize>, Errors> {
        let res: Vec<_> = self
            .into_iter()
            .scan(vec![0; self.no_dim], |state, val| {
                let keep = val != NumCast::from(0).unwrap();
                let ret = state.clone();
                increment_dim_index_unchecked(state, 0, &self.dims, &self.strides);

                Some((ret, keep))
            })
            .filter(|(_, x)| *x)
            .map(|(x, _)| x)
            .flatten()
            .collect();

        Tensor::from_slice_and_dims(&res, &[res.len() / self.no_dim, self.no_dim])
    }

    /// Alias to [argwhere](Tensor::argwhere)
    pub fn nonzero(&self) -> Result<Tensor<usize>, Errors> {
        self.argwhere()
    }
}

impl<T: Copy> Tensor<T> {
    /// Returns a Tensor which is the concatenation of `self` and `other` along dimension `dim`
    ///
    /// The 2 tensors must have equal number of dimensions and the dimensions can differ only at
    /// concatenation dimension
    ///
    /// # Arguments
    /// * other - The other tensor to concatenate with self
    /// * dim - The dimension index to concatenate on
    ///
    /// # Examples
    /// ```rust
    /// let t1 =
    ///     Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[3, 4]).unwrap();
    /// let t2 = Tensor::from_slice_and_dims(&[13, 14, 15, 16, 17, 18], &[3, 2]).unwrap();
    /// let res = t1.cat(&t2, 1).unwrap();
    /// // res = [[1, 2, 3, 4, 13, 14], [5, 6, 7, 8, 15, 16], [9, 10, 11, 12, 17, 18]]
    /// ```
    pub fn cat(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, Errors> {
        if self.no_dim != other.no_dim {
            return Err(Errors::InputError(format!(
                "Concat Error: Expected number of dimensions for both tensors to match, found {} != {}",
                self.no_dim, other.no_dim
            )));
        }

        if dim >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Concat Error: Expected concat dimension < number of dimensions, found {} >= {}",
                dim, self.no_dim
            )));
        }

        if self
            .dims
            .iter()
            .zip(other.dims.iter())
            .enumerate()
            .filter(|(_, (&c_d, &o_d))| c_d != o_d)
            .any(|(idx, (_, _))| idx != dim)
        {
            return Err(Errors::InputError(
                "Concat Error: Expected dimensions of both tensors to match except the concat dimension".to_string(),
            ));
        }

        let mut dims = self.dims.clone();
        dims[dim] += other.dims[dim];
        let strides = new_strides_from_dim(&dims);

        let take_self: usize = self.dims.iter().skip(dim).product();
        let take_other: usize = other.dims.iter().skip(dim).product();

        let mut self_it = self.into_iter();
        let mut other_it = other.into_iter();
        let mut storage_vec: Vec<T> = vec![];
        while storage_vec.len() < self.no_el + other.no_el {
            for _ in 0..take_self {
                if let Some(val) = self_it.next() {
                    storage_vec.push(val);
                } else {
                    break;
                }
            }

            for _ in 0..take_other {
                if let Some(val) = other_it.next() {
                    storage_vec.push(val);
                } else {
                    break;
                }
            }
        }

        let storage = Rc::new(RefCell::new(TensorStorage::from_slice(&storage_vec)));
        return Ok(Tensor::new_unchecked(storage, 0, &dims, &strides));
    }

    /// Alias to [cat](Tensor::cat)
    pub fn concat(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, Errors> {
        self.cat(other, dim)
    }

    /// Alias to [cat](Tensor::cat)
    pub fn concatenate(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, Errors> {
        self.cat(other, dim)
    }

    /// Split tensors into `cnt` number of pieces along dimension `dim` and return the collection
    /// of tensors
    ///
    /// # Arguments
    /// * cnt - The number of chunks
    /// * dim - The dimension to chunk on
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(0, 11, 1).unwrap();
    /// let res = t.chunk(6, 0).unwrap();
    /// // res: Vec<Tensor<i32>> = vec![[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]
    /// ```
    pub fn chunk(&self, cnt: usize, dim: usize) -> Result<Vec<Tensor<T>>, Errors> {
        if cnt == 0 {
            return Err(Errors::InputError(
                "Chunk Error: Expected cnt > 0".to_string(),
            ));
        }

        if dim >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Chunk Error: Expected chunk dimension < number of dimensions, found {} >= {}",
                dim, self.no_dim
            )));
        }

        let per = (self.dims[dim] + cnt - 1) / cnt;
        let no = (self.dims[dim] + per - 1) / per;
        let mut slice_index: Vec<_> = self.dims.iter().map(|&dim_sz| 0..dim_sz).collect();
        return Ok((0..no)
            .map(|rng| {
                slice_index[dim] = (per * rng)..(per * rng + per).min(self.dims[dim]);
                self.slice_unchecked(&slice_index)
            })
            .collect::<Vec<_>>());
    }

    /// Return tensor with values according to the index in the corresponding element in
    /// `indices` along dimension `dim`
    ///
    /// # Arguments
    /// * dim - The dimension to gather on
    /// * indices - The indices tensor
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 5, 1).unwrap().reshape(&[2, 2]).unwrap();
    /// let ind = Tensor::from_slice_and_dims(&[0, 0, 1, 0], &[2, 2]).unwrap();
    /// let gather = t.gather(1, ind);
    /// // gather = [[1, 1], [4, 3]]
    /// ```
    pub fn gather(&self, dim: usize, indices: &Tensor<usize>) -> Result<Tensor<T>, Errors> {
        if self.dims != indices.dims {
            return Err(Errors::InputError(
                "Gather Error: Expected Index Tensor to have same dimensions as self".to_string(),
            ));
        }

        if dim >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Gather Error: Expected gather dimension < number of dimensions, found {} >= {}",
                dim, self.no_dim
            )));
        }

        if let Some(val) = indices.into_iter().find(|&val| val >= self.dims[dim]) {
            return Err(Errors::OutOfBounds {
                expected: self.dims[dim],
                found: val,
                axis: dim,
            });
        }

        let res: Vec<_> = self
            .into_iter()
            .scan((vec![0; self.no_dim], self.offset), |state, _| {
                let val = indices.at_unchecked(&state.0);
                let prev_val = state.0[dim];

                state.0[dim] = val;
                state.1 -= self.strides[dim] * prev_val;
                state.1 += self.strides[dim] * val;
                let ret = self.storage.borrow().get_unchecked(state.1);
                state.1 -= self.strides[dim] * val;
                state.1 += self.strides[dim] * prev_val;
                state.0[dim] = prev_val;

                (state.1, _) =
                    increment_dim_index_unchecked(&mut state.0, state.1, &self.dims, &self.strides);

                Some(ret)
            })
            .collect();

        Tensor::from_slice_and_dims(&res, &self.dims)
    }

    /// Returns a 1D Tensor which picks elements from `self` if the corresponding element in
    /// `mask` in true
    ///
    /// The dimensions of the specified tensor must match the dimensions of self
    ///
    /// # Arguments
    /// * pick - The mask tensor
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::from_slice_and_dims(
    ///     &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ///     &[3, 5],
    /// )
    /// .unwrap();
    /// let pick = Tensor::from_slice_and_dims(
    ///     &[
    ///         true, true, false, true, false, false, false, true, false, false, false, false,
    ///         false, false, false,
    ///     ],
    ///     &[3, 5],
    /// )
    /// .unwrap();
    /// let res = t.masked_select(&pick).unwrap();
    /// // res = [1, 2, 4, 8]
    ///```
    pub fn masked_select(&self, pick: &Tensor<bool>) -> Result<Tensor<T>, Errors> {
        if self.dims != pick.dims {
            return Err(Errors::InputError(
                "Masked Select Error: Expected Index Tensor to have same dimensions as self"
                    .to_string(),
            ));
        }

        let res: Vec<_> = self
            .into_iter()
            .zip(pick.into_iter())
            .filter(|(_, x)| *x)
            .map(|(x, _)| x)
            .collect();

        Tensor::from_slice_and_dims(&res, &[res.len()])
    }

    /// Returns a 1D Tensor which picks elements from `self` if the function `f` returns true
    /// given the element
    ///
    /// # Arguments
    /// * f - The selector function
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 11, 1).unwrap();
    /// let res = t.fn_select(|val, _| val % 2 == 0).unwrap();
    /// // res = [2, 4, 6, 8, 10]
    /// ```
    pub fn fn_select<F: Fn(T, &[usize]) -> bool>(&self, f: F) -> Result<Tensor<T>, Errors> {
        let res: Vec<_> = self
            .into_iter()
            .scan(vec![0; self.no_dim], |state, val| {
                let keep = f(val, state);
                increment_dim_index_unchecked(state, 0, &self.dims, &self.strides);

                Some((val, keep))
            })
            .filter(|(_, x)| *x)
            .map(|(x, _)| x)
            .collect();

        Tensor::from_slice_and_dims(&res, &[res.len()])
    }

    /// Returns a slice of `self` along dimension `dim` from `st` of length `len`
    ///
    /// # Arguments
    /// * dim - Dimension to narrow
    /// * st - Start index
    /// * len - Length of slice
    ///
    /// # Examples
    /// ```rust
    /// let t =
    ///     Tensor::arange(1, 13, 1).unwrap().reshape(&[3, 4]);
    /// let res = t.narrow(1, 1, 3).unwrap();
    /// // res = [[2, 3, 4], [6, 7, 8], [10, 11, 12]]
    /// ```
    pub fn narrow(&self, dim: usize, st: usize, len: usize) -> Result<Tensor<T>, Errors> {
        if dim >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Narrow Error: Expected concat dimension < number of dimensions, found {} >= {}",
                dim, self.no_dim
            )));
        }

        if len == 0 {
            return Err(Errors::InputError(
                "Narrow Error: Expected len > 0".to_string(),
            ));
        }

        if st + len > self.dims[dim] {
            return Err(Errors::OutOfBounds {
                expected: self.dims[dim],
                found: st + len - 1,
                axis: dim,
            });
        }

        let rngs: Vec<_> = self
            .dims
            .iter()
            .enumerate()
            .map(
                |(i, &dim_sz)| {
                    if i == dim {
                        st..st + len
                    } else {
                        0..dim_sz
                    }
                },
            )
            .collect();

        Ok(self.slice_unchecked(&rngs))
    }

    /// Return tensor with same data as self but with dimensions `new_dims`. The output tensor can
    /// possibly be a view of `self`
    ///
    /// # Arguments
    /// * new_dims - The new dimensions
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 13).unwrap();
    /// let res = t.reshape(&[3, 4]);
    /// // res = [[1, 2, 3, 4], [5, 6, 7, 8], [11, 12, 13, 14]]
    /// ```
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>, Errors> {
        if self.no_el != new_dims.iter().product() {
            return Err(Errors::InputError(format!(
                "Reshape Error: Invalid dimensions for current tensor, dims = {:?} is not valid for tensor with {} elements",
                new_dims, self.no_el
            )));
        }

        if self.dims == new_dims {
            return Ok(self.clone());
        }

        if self.is_view() {
            Tensor::from_slice_and_dims(&self.into_iter().collect::<Vec<_>>(), new_dims)
        } else {
            Tensor::new(
                Rc::clone(&self.storage),
                self.offset,
                &new_dims,
                &new_strides_from_dim(new_dims),
            )
        }
    }

    /// Returns a tensor with same data as `self` but with it's dimension permuted by the
    /// permutation `permutation`
    ///
    /// # Arguments
    /// * permutation - The permutation to permute the dimensions to. Must be a permutation from 0
    /// to number of dimensions - 1
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 11, 1).unwrap().reshape(&[5, 2]).unwrap();
    /// let res = t.permute(&[1, 0]).unwrap();
    /// // res = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
    /// ```
    pub fn permute(&self, permutation: &[usize]) -> Result<Tensor<T>, Errors> {
        if permutation.len() != self.no_dim {
            return Err(Errors::InputError(format!(
                "Permute Error: Expected permutation len = no. of dimensions, found {} != {}",
                permutation.len(),
                self.no_dim
            )));
        }

        if permutation.iter().any(|&x| x >= self.no_dim) {
            return Err(Errors::InputError(format!(
                "Permute Error: Expected valid permutation, found: {:?}",
                permutation
            )));
        }

        let mut seen = vec![false; self.no_dim];
        permutation.iter().for_each(|&x| seen[x] = true);

        if seen.iter().any(|x| !x) {
            return Err(Errors::InputError(format!(
                "Permute Error: Expected valid permutation, found: {:?}",
                permutation
            )));
        }

        let new_dims: Vec<_> = permutation.iter().map(|&v| self.dims[v]).collect();
        let new_strides: Vec<_> = permutation.iter().map(|&v| self.strides[v]).collect();
        Tensor::new(
            Rc::clone(&self.storage),
            self.offset,
            &new_dims,
            &new_strides,
        )
    }

    /// Return the tensor obtained by swapping dimensions `dim1` and `dim2` of `self`
    ///
    /// # Arguments
    /// * dim1 - Dimension 1
    /// * dim2 - Dimension 2
    ///
    /// # Examples
    /// ```rust
    /// let t = Tensor::arange(1, 11, 1)
    ///     .unwrap()
    ///     .reshape(&[5, 2])
    ///     .unwrap()
    /// let res = t.transpose(0, 1).unwrap();
    /// // res = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
    /// ```
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Tensor<T>, Errors> {
        if dim1 >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Transpose Error: Expected transpose dimension < number of dimensions, found {} >= {}",
                dim1, self.no_dim
            )));
        }

        if dim2 >= self.no_dim {
            return Err(Errors::InputError(format!(
                "Transpose Error: Expected transpose dimension < number of dimensions, found {} >= {}",
                dim2, self.no_dim
            )));
        }

        let mut new_dims = self.dims.clone();
        let mut new_strides = self.strides.clone();

        let tmp = new_dims[dim1];
        new_dims[dim1] = new_dims[dim2];
        new_dims[dim2] = tmp;

        let tmp = new_strides[dim1];
        new_strides[dim1] = new_strides[dim2];
        new_strides[dim2] = tmp;

        Tensor::new(
            Rc::clone(&self.storage),
            self.offset,
            &new_dims,
            &new_strides,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::tensor::tensor::Tensor, utils::errors::Errors};

    #[test]
    fn argwhere() {
        let t = Tensor::from_slice(&[1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0]).unwrap();
        let res = t.argwhere().unwrap();
        assert_eq!(res.dims, &[5, 1]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), vec![0, 2, 3, 4, 7]);

        let t = Tensor::<f64>::zeros(&[2, 3]).unwrap();
        t.upd(&[0, 0], 5.0).unwrap();
        t.upd(&[0, 2], 5.0).unwrap();
        t.upd(&[1, 1], 5.0).unwrap();
        let res = t.argwhere().unwrap();
        assert_eq!(res.dims, &[3, 2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[0, 0, 0, 2, 1, 1]);

        assert!(
            match Tensor::<f32>::zeros(&[1, 2, 3, 4, 5]).unwrap().argwhere() {
                Ok(_) => false,
                Err(e) => match e {
                    Errors::EmptyTensor => true,
                    _ => false,
                },
            }
        );
    }

    #[test]
    fn cat() {
        let t1 = Tensor::arange(1, 5, 1).unwrap();
        let t2 = Tensor::arange(5, 10, 1).unwrap();
        let res = t1.cat(&t2, 0).unwrap();
        assert_eq!(res.dims, &[9]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9]
        );

        let t1 =
            Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[3, 4]).unwrap();
        let t2 = Tensor::from_slice_and_dims(&[13, 14, 15, 16, 17, 18], &[3, 2]).unwrap();
        let res = t1.cat(&t2, 1).unwrap();
        assert_eq!(res.dims, &[3, 6]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            &[1, 2, 3, 4, 13, 14, 5, 6, 7, 8, 15, 16, 9, 10, 11, 12, 17, 18]
        );

        let t1 = Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[2, 2, 3])
            .unwrap();
        let t2 = Tensor::from_slice_and_dims(
            &[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            &[2, 2, 3],
        )
        .unwrap();
        let res = t1.cat(&t2, 0).unwrap();
        assert_eq!(res.dims, &[4, 2, 3]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            (1..25).collect::<Vec<_>>()
        );
        let res = t1.cat(&t2, 1).unwrap();
        assert_eq!(res.dims, &[2, 4, 3]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            &[
                1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23,
                24
            ]
        );
        let res = t1.cat(&t2, 2).unwrap();
        assert_eq!(res.dims, &[2, 2, 6]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            &[
                1, 2, 3, 13, 14, 15, 4, 5, 6, 16, 17, 18, 7, 8, 9, 19, 20, 21, 10, 11, 12, 22, 23,
                24
            ]
        );

        assert!(match t1.cat(&t2, 3) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        let t2 = Tensor::from_slice_and_dims(&[13, 14, 15, 16], &[2, 2, 1]).unwrap();
        assert!(match t1.cat(&t2, 3) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        let t2 = Tensor::from_slice_and_dims(&[13, 14, 15, 16], &[2, 2]).unwrap();
        assert!(match t1.cat(&t2, 3) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });
    }

    #[test]
    fn chunk() {
        let t = Tensor::arange(0, 11, 1).unwrap();
        let res = t.chunk(6, 0).unwrap();
        assert_eq!(res.len(), 6);
        assert_eq!(res[0].into_iter().collect::<Vec<_>>(), &[0, 1]);
        assert_eq!(res[1].into_iter().collect::<Vec<_>>(), &[2, 3]);
        assert_eq!(res[2].into_iter().collect::<Vec<_>>(), &[4, 5]);
        assert_eq!(res[3].into_iter().collect::<Vec<_>>(), &[6, 7]);
        assert_eq!(res[4].into_iter().collect::<Vec<_>>(), &[8, 9]);
        assert_eq!(res[5].into_iter().collect::<Vec<_>>(), &[10]);

        let t = Tensor::arange(0, 12, 1).unwrap();
        let res = t.chunk(6, 0).unwrap();
        assert_eq!(res.len(), 6);
        assert_eq!(res[0].into_iter().collect::<Vec<_>>(), &[0, 1]);
        assert_eq!(res[1].into_iter().collect::<Vec<_>>(), &[2, 3]);
        assert_eq!(res[2].into_iter().collect::<Vec<_>>(), &[4, 5]);
        assert_eq!(res[3].into_iter().collect::<Vec<_>>(), &[6, 7]);
        assert_eq!(res[4].into_iter().collect::<Vec<_>>(), &[8, 9]);
        assert_eq!(res[5].into_iter().collect::<Vec<_>>(), &[10, 11]);

        let t = Tensor::arange(0, 13, 1).unwrap();
        let res = t.chunk(6, 0).unwrap();
        assert_eq!(res.len(), 5);
        assert_eq!(res[0].into_iter().collect::<Vec<_>>(), &[0, 1, 2]);
        assert_eq!(res[1].into_iter().collect::<Vec<_>>(), &[3, 4, 5]);
        assert_eq!(res[2].into_iter().collect::<Vec<_>>(), &[6, 7, 8]);
        assert_eq!(res[3].into_iter().collect::<Vec<_>>(), &[9, 10, 11]);
        assert_eq!(res[4].into_iter().collect::<Vec<_>>(), &[12]);

        assert!(match t.chunk(0, 0) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        })
    }

    #[test]
    fn gather() {
        let t = Tensor::from_slice_and_dims(&[1, 2, 3, 4], &[2, 2]).unwrap();
        let ind = Tensor::from_slice_and_dims(&[0, 1, 1, 0], &[2, 2]).unwrap();
        let res = t.gather(0, &ind).unwrap();
        assert_eq!(res.dims, &[2, 2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 4, 3, 2]);

        let t = Tensor::from_slice_and_dims(&[1, 2, 3, 4], &[2, 2]).unwrap();
        let ind = Tensor::from_slice_and_dims(&[0, 0, 1, 0], &[2, 2]).unwrap();
        let res = t.gather(1, &ind).unwrap();
        assert_eq!(res.dims, &[2, 2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 1, 4, 3]);

        let tt = Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3]).unwrap();
        let t = tt.slice(&[0..2, 0..2]).unwrap();
        let ind = Tensor::from_slice_and_dims(&[0, 1, 1, 0], &[2, 2]).unwrap();
        let res = t.gather(0, &ind).unwrap();
        assert_eq!(res.dims, &[2, 2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 5, 4, 2]);

        let ind = Tensor::from_slice_and_dims(&[0, 0, 1, 0], &[2, 2]).unwrap();
        let res = t.gather(2, &ind);
        assert!(match res {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        let ind = Tensor::from_slice_and_dims(&[0, 0, 1, 0, 0], &[5, 1]).unwrap();
        let res = t.gather(2, &ind);
        assert!(match res {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        let ind = Tensor::from_slice_and_dims(&[0, 2, 1, 0], &[2, 2]).unwrap();
        let res = t.gather(0, &ind);
        assert!(match res {
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
    fn masked_select() {
        let t = Tensor::arange(1, 11, 1).unwrap();
        let pick = Tensor::from_slice_and_dims(
            &[
                true, true, false, false, true, true, false, false, true, true,
            ],
            &[10],
        )
        .unwrap();
        let res = t.masked_select(&pick).unwrap();
        assert_eq!(res.dims, &[6]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 2, 5, 6, 9, 10]);

        let t = Tensor::from_slice_and_dims(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            &[3, 5],
        )
        .unwrap();
        let pick = Tensor::from_slice_and_dims(
            &[
                true, true, false, true, false, false, false, true, false, false, false, false,
                false, false, false,
            ],
            &[3, 5],
        )
        .unwrap();
        let res = t.masked_select(&pick).unwrap();
        assert_eq!(res.dims, &[4]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 2, 4, 8]);

        let t = t.slice(&[0..2, 0..2]).unwrap();
        let pick = Tensor::from_slice_and_dims(&[true, false, false, true], &[2, 2]).unwrap();
        let res = t.masked_select(&pick).unwrap();
        assert_eq!(res.dims, &[2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 7]);

        let pick =
            Tensor::from_slice_and_dims(&[true, false, false, true, true, false], &[2, 3]).unwrap();
        assert!(match t.masked_select(&pick) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        let pick = Tensor::from_slice_and_dims(&[false, false, false, false], &[2, 2]).unwrap();
        assert!(match t.masked_select(&pick) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });
    }

    #[test]
    fn fn_select() {
        let t = Tensor::arange(1, 11, 1).unwrap();
        let res = t.fn_select(|val, _| val % 2 == 0).unwrap();
        assert_eq!(res.dims, &[5]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[2, 4, 6, 8, 10]);

        let t = Tensor::from_slice_and_dims(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            &[3, 5],
        )
        .unwrap();
        let res = t.fn_select(|val: i32, _| val.count_ones() == 1).unwrap();
        assert_eq!(res.dims, &[4]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 2, 4, 8]);

        let t = t.slice(&[0..2, 0..2]).unwrap();
        let res = t
            .fn_select(|_, idx| idx == &[0, 0] || idx == &[1, 1])
            .unwrap();
        assert_eq!(res.dims, &[2]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[1, 7]);

        assert!(match t.fn_select(|_, _| false) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::EmptyTensor => true,
                _ => false,
            },
        });
    }

    #[test]
    pub fn narrow() {
        let t = Tensor::arange(1, 11, 1).unwrap();
        let res = t.narrow(0, 2, 3).unwrap();
        assert_eq!(res.dims, &[3]);
        assert_eq!(res.into_iter().collect::<Vec<_>>(), &[3, 4, 5]);

        let t =
            Tensor::from_slice_and_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[3, 4]).unwrap();
        let res = t.narrow(1, 1, 3).unwrap();
        assert_eq!(res.dims, &[3, 3]);
        assert_eq!(
            res.into_iter().collect::<Vec<_>>(),
            &[2, 3, 4, 6, 7, 8, 10, 11, 12]
        );

        assert!(match t.narrow(2, 0, 1) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        assert!(match t.narrow(1, 0, 0) {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        assert!(match t.narrow(1, 0, 5) {
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
    fn reshape() {
        let t = Tensor::arange(1, 11, 1).unwrap().reshape(&[2, 5]).unwrap();
        assert_eq!(t.dims, vec![2, 5]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );

        let t = Tensor::arange(0, 27, 1)
            .unwrap()
            .reshape(&[3, 3, 3])
            .unwrap();
        assert_eq!(t.dims, vec![3, 3, 3]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            (0..27).collect::<Vec<_>>()
        );

        let t = Tensor::linspace(1.0, 10.0, 9)
            .unwrap()
            .slice(&[0..3])
            .unwrap()
            .reshape(&[3, 1, 1])
            .unwrap();
        assert_eq!(t.dims, vec![3, 1, 1]);
        assert_eq!(t.into_iter().collect::<Vec<_>>(), vec![1.0, 2.125, 3.25]);

        assert!(
            match Tensor::linspace(1.0, 10.0, 100).unwrap().reshape(&[2, 2]) {
                Ok(_) => false,
                Err(e) => match e {
                    Errors::InputError(_) => true,
                    _ => false,
                },
            }
        );
    }

    #[test]
    fn permute() {
        let t = Tensor::arange(1, 11, 1)
            .unwrap()
            .reshape(&[5, 2])
            .unwrap()
            .permute(&[1, 0])
            .unwrap();
        assert_eq!(t.dims, &[2, 5]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            &[1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        );

        let t = Tensor::linspace(0.0, 9.5, 20)
            .unwrap()
            .reshape(&[2, 2, 5])
            .unwrap()
            .permute(&[2, 0, 1])
            .unwrap();
        assert_eq!(t.dims, &[5, 2, 2]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            &[
                0.0000, 2.5000, 5.0000, 7.5000, 0.5000, 3.0000, 5.5000, 8.0000, 1.0000, 3.5000,
                6.0000, 8.5000, 1.5000, 4.0000, 6.5000, 9.0000, 2.0000, 4.5000, 7.0000, 9.5000
            ]
        );

        assert!(match Tensor::linspace(0.0, 9.5, 20)
            .unwrap()
            .reshape(&[2, 2, 5])
            .unwrap()
            .permute(&[2, 0, 1, 3])
        {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        assert!(match Tensor::linspace(0.0, 9.5, 20)
            .unwrap()
            .reshape(&[2, 2, 5])
            .unwrap()
            .permute(&[3, 0, 1])
        {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        assert!(match Tensor::linspace(0.0, 9.5, 20)
            .unwrap()
            .reshape(&[2, 2, 5])
            .unwrap()
            .permute(&[2, 2, 1])
        {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });
    }

    #[test]
    fn transpose() {
        let t = Tensor::arange(1, 11, 1)
            .unwrap()
            .reshape(&[5, 2])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(t.dims, &[2, 5]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            &[1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        );

        let t = Tensor::linspace(0.0, 9.5, 20)
            .unwrap()
            .reshape(&[2, 2, 5])
            .unwrap()
            .transpose(0, 2)
            .unwrap();
        assert_eq!(t.dims, &[5, 2, 2]);
        assert_eq!(
            t.into_iter().collect::<Vec<_>>(),
            &[
                0.0000, 5.0000, 2.5000, 7.5000, 0.5000, 5.5000, 3.0000, 8.0000, 1.0000, 6.0000,
                3.5000, 8.5000, 1.5000, 6.5000, 4.0000, 9.0000, 2.0000, 7.0000, 4.5000, 9.5000
            ]
        );

        assert!(match Tensor::arange(1, 11, 1)
            .unwrap()
            .reshape(&[5, 2])
            .unwrap()
            .transpose(0, 2)
        {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });

        assert!(match Tensor::arange(1, 11, 1)
            .unwrap()
            .reshape(&[5, 2])
            .unwrap()
            .transpose(2, 1)
        {
            Ok(_) => false,
            Err(e) => match e {
                Errors::InputError(_) => true,
                _ => false,
            },
        });
    }
}
