use super::errors::Errors;

pub(crate) fn dim_index_to_storage_index(
    index: &[usize],
    offset: usize,
    dims: &[usize],
    strides: &[usize],
) -> Result<usize, Errors> {
    if index.len() != dims.len() {
        Err(Errors::InvalidIndexSize {
            expected: dims.len(),
            found: index.len(),
        })
    } else if let Some(idx) = index
        .iter()
        .zip(dims.iter())
        .position(|(&idx, &sz)| idx >= sz)
    {
        Err(Errors::OutOfBounds {
            expected: dims[idx],
            found: index[idx],
            axis: idx,
        })
    } else {
        Ok(dim_index_to_storage_index_unchecked(index, offset, strides))
    }
}

pub(crate) fn dim_index_to_storage_index_unchecked(
    index: &[usize],
    offset: usize,
    strides: &[usize],
) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .fold(0, |res, (idx, stride)| res + idx * stride)
        + offset
}

pub(crate) fn increment_dim_index(
    index: &mut [usize],
    storage_index: usize,
    dims: &[usize],
    strides: &[usize],
) -> Result<(usize, bool), Errors> {
    if index.len() != dims.len() {
        return Err(Errors::InvalidIndexSize {
            expected: dims.len(),
            found: index.len(),
        });
    }

    if let Some(idx) = index
        .iter()
        .zip(dims.iter())
        .position(|(&idx, &sz)| idx >= sz)
    {
        return Err(Errors::OutOfBounds {
            expected: dims[idx],
            found: index[idx],
            axis: idx,
        });
    }

    Ok(increment_dim_index_unchecked(
        index,
        storage_index,
        dims,
        strides,
    ))
}

pub(crate) fn increment_dim_index_unchecked(
    index: &mut [usize],
    mut storage_index: usize,
    dims: &[usize],
    strides: &[usize],
) -> (usize, bool) {
    if index
        .iter()
        .zip(dims.iter())
        .all(|(&idx, &sz)| idx == sz - 1)
    {
        return (storage_index, true);
    }

    let idx = index.len()
        - index
            .iter()
            .rev()
            .zip(dims.iter().rev())
            .take_while(|(&idx, &sz)| idx == sz - 1)
            .count()
        - 1;
    index[idx] += 1;
    storage_index += strides[idx];

    index.iter().enumerate().skip(idx + 1).for_each(|(i, &x)| {
        storage_index -= strides[i] * x;
    });

    index.iter_mut().skip(idx + 1).for_each(|x| *x = 0);
    (storage_index, false)
}
