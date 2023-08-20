pub(crate) fn new_strides_from_dim(dims: &[usize]) -> Vec<usize> {
    let mut rev_strides: Vec<_> = dims
        .iter()
        .rev()
        .scan(1, |state, &dim| {
            let tmp = Some(*state);
            *state *= dim;
            tmp
        })
        .collect();
    rev_strides.reverse();
    rev_strides
}
