"""
Based on https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/utils/torch_utils.py
and extended with further useful functions.
"""
import logging
from typing import Any

import numpy as np
import torch as ch


def sqrtm_newton(x: ch.Tensor, **kwargs: Any):
    """
    From: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    License: MIT

    Compute the Sqrt of a matrix based on Newton-Schulz algorithm
    """
    num_iters = kwargs.get("num_iters") or 10

    batch_size = x.shape[0]
    dim = x.shape[-1]
    dtype = x.dtype

    normA = x.pow(2).sum(dim=1).sum(dim=1).sqrt()
    Y = x / normA.view(batch_size, 1, 1).expand_as(x)
    I = 3.0 * ch.eye(dim, dtype=dtype)
    Z = ch.eye(dim, dtype=dtype)
    for i in range(num_iters):
        T = 0.5 * (I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sA = Y * normA.sqrt().view(batch_size, 1, 1).expand_as(x)
    return sA


def sqrtm(x: ch.Tensor):
    """
    Compute the Sqrt of a matrix based on eigen decomposition. Assumes the matrix is symmetric PSD.

    Args:
        x: data

    Returns:
        matrix sqrt of x
    """
    eigvals, eigvecs = x.symeig(eigenvectors=True, upper=False)
    return eigvecs @ (ch.sqrt(eigvals)).diag_embed(0, -2, -1) @ eigvecs.permute(0, 2, 1)


def torch_batched_trace(x) -> ch.Tensor:
    """
    Compute trace in n,m of batched matrix X
    Args:
        x: matrix with shape [a,..., l, n, m]

    Returns: trace with shape [a,..., l]

    """
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def torch_batched_trace_square(x):
    """
    Compute trace in n,m of squared batched matrix XX^{T}
    Args:
        x: matrix with shape [a,..., l, n, m]

    Returns: trace with shape [a,...l]

    """
    n = x.size(-1)
    m = x.size(-2)
    flat_trace = x.reshape(-1, m * n).square().sum(-1)
    return flat_trace.reshape(x.shape[:-2])


def tensorize(x, cpu=True, dtype=ch.float32):
    """
    Utility function for turning arrays into tensors
    Args:
        x: data
        cpu: Whether to generate a CPU or GPU tensor
        dtype: dtype of tensor

    Returns:
        gpu/cpu tensor of x with specified dtype
    """
    return cpu_tensorize(x, dtype) if cpu else gpu_tensorize(x, dtype)


def gpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cuda tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        gpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype).cuda()


def cpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cpu tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        cpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype)


def to_gpu(x):
    """
    Utility function for turning tensors into gpu tensors
    Args:
        x: data

    Returns:
        gpu tensor of x
    """
    return x.cuda()


def get_numpy(x):
    """
    Convert torch tensor to numpy
    Args:
        x: torch.Tensor

    Returns:
        numpy tensor of x

    """
    return x.cpu().detach().numpy()


def select_batch(index, *args) -> list:
    """
    For each argument select the value at index.
    Args:
        index: index of values to select
        *args: data

    Returns:
        list of indexed value
    """
    return [v[index] for v in args]


def flatten_batch(x):
    """
        flatten axes 0 and 1
    Args:
        x: tensor to flatten

    Returns:
        flattend tensor version of x
    """

    s = x.shape
    return x.contiguous().view([s[0] * s[1], *s[2:]])


def generate_minibatches(n, n_minibatches):
    """
    Generate n_minibatches sets of indices for N data points.  
    Args:
        n: total number of data points
        n_minibatches: how many minibatches to generate

    Returns:
        np.ndarray of minibatched indices
    """
    state_indices = np.arange(n)
    np.random.shuffle(state_indices)
    return np.array_split(state_indices, n_minibatches)


def fill_triangular(x, upper=False):
    """
    Adapted from:
    https://github.com/tensorflow/probability/blob/c833ee5cd9f60f3257366b25447b9e50210b0590/tensorflow_probability/python/math/linalg.py#L787
    License: Apache-2.0

    Creates a (batch of) triangular matrix from a vector of inputs.

    Created matrix can be lower- or upper-triangular. (It is more efficient to
    create the matrix as upper or lower, rather than transpose.)

    Triangular matrix elements are filled in a clockwise spiral. See example,
    below.

    If `x.shape` is `[b1, b2, ..., bB, dim]` then the output shape is
    `[b1, b2, ..., bB, n, n]` where `n` is such that `dim = n(n+1)/2`, i.e.,
    `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

    Example:

    ```python
    fill_triangular([1, 2, 3, 4, 5, 6])
    # ==> [[4, 0, 0],
    #      [6, 5, 0],
    #      [3, 2, 1]]

    fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
    # ==> [[1, 2, 3],
    #      [0, 5, 6],
    #      [0, 0, 4]]
    ```

    The key trick is to create an upper triangular matrix by concatenating `x`
    and a tail of itself, then reshaping.

    Suppose that we are filling the upper triangle of an `n`-by-`n` matrix `M`
    from a vector `x`. The matrix `M` contains n**2 entries total. The vector `x`
    contains `n * (n+1) / 2` entries. For concreteness, we'll consider `n = 5`
    (so `x` has `15` entries and `M` has `25`). We'll concatenate `x` and `x` with
    the first (`n = 5`) elements removed and reversed:

    ```python
    x = np.arange(15) + 1
    xc = np.concatenate([x, x[5:][::-1]])
    # ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13,
    #            12, 11, 10, 9, 8, 7, 6])

    # (We add one to the arange result to disambiguate the zeros below the
    # diagonal of our upper-triangular matrix from the first entry in `x`.)

    # Now, when reshapedlay this out as a matrix:
    y = np.reshape(xc, [5, 5])
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 6,  7,  8,  9, 10],
    #            [11, 12, 13, 14, 15],
    #            [15, 14, 13, 12, 11],
    #            [10,  9,  8,  7,  6]])

    # Finally, zero the elements below the diagonal:
    y = np.triu(y, k=0)
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 0,  7,  8,  9, 10],
    #            [ 0,  0, 13, 14, 15],
    #            [ 0,  0,  0, 12, 11],
    #            [ 0,  0,  0,  0,  6]])
    ```

    From this example we see that the resuting matrix is upper-triangular, and
    contains all the entries of x, as desired. The rest is details:

    - If `n` is even, `x` doesn't exactly fill an even number of rows (it fills
      `n / 2` rows and half of an additional row), but the whole scheme still
      works.
    - If we want a lower triangular matrix instead of an upper triangular,
      we remove the first `n` elements from `x` rather than from the reversed
      `x`.

    For additional comparisons, a pure numpy version of this function can be found
    in `distribution_util_test.py`, function `_fill_triangular`.

    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
      tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

    Raises:
      ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    m = np.int32(x.shape[-1])
    # Formula derived by solving for n: m = n(n+1)/2.
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
    n = np.int32(n)
    new_shape = x.shape[:-1] + (n, n)

    ndims = len(x.shape)
    if upper:
        x_list = [x, ch.flip(x[..., n:], dims=[ndims - 1])]
    else:
        x_list = [x[..., n:], ch.flip(x, dims=[ndims - 1])]

    x = ch.cat(x_list, dim=-1).reshape(new_shape)
    x = ch.triu(x) if upper else ch.tril(x)
    return x


def fill_triangular_inverse(x, upper=False):
    """
    Adapted from:
    https://github.com/tensorflow/probability/blob/c833ee5cd9f60f3257366b25447b9e50210b0590/tensorflow_probability/python/math/linalg.py#L937
    License: Apache-2.0

    Creates a vector from a (batch of) triangular matrix.

    The vector is created from the lower-triangular or upper-triangular portion
    depending on the value of the parameter `upper`.

    If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
    `[b1, b2, ..., bB, dim]` where `dim = n (n + 1) / 2`.

    Example:

    ```python
    fill_triangular_inverse(
      [[4, 0, 0],
       [6, 5, 0],
       [3, 2, 1]])

    # ==> [1, 2, 3, 4, 5, 6]

    fill_triangular_inverse(
      [[1, 2, 3],
       [0, 5, 6],
       [0, 0, 4]], upper=True)

    # ==> [1, 2, 3, 4, 5, 6]
    ```

    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
      flat_tril: (Batch of) vector-shaped `Tensor` representing vectorized lower
        (or upper) triangular elements from `x`.
    """

    n = np.int32(x.shape[-1])
    m = np.int32((n * (n + 1)) // 2)

    ndims = len(x.shape)
    if upper:
        initial_elements = x[..., 0, :]
        triangular_part = x[..., 1:, :]
    else:
        initial_elements = ch.flip(x[..., -1, :], dims=[ndims - 2])
        triangular_part = x[..., :-1, :]

    rotated_triangular_portion = ch.flip(ch.flip(triangular_part, dims=[ndims - 1]), dims=[ndims - 2])
    consolidated_matrix = triangular_part + rotated_triangular_portion

    end_sequence = consolidated_matrix.reshape(x.shape[:-2] + (n * (n - 1),))

    y = ch.cat([initial_elements, end_sequence[..., :m - n]], dim=-1)
    return y


def diag_bijector(f: callable, x):
    """
    Apply transformation f(x) on the diagonal of a batched matrix.
    Args:
        f: callable to apply
        x: data

    Returns:
        transformed matrix x
    """
    transformed = x.tril(-1) + f(x.diagonal(dim1=-2, dim2=-1)).diag_embed() + x.triu(1)
    return transformed.squeeze(dim=0)


def inverse_softplus(x):
    """
    x = inverse_softplus(softplus(x))
    Adapted from https://github.com/tensorflow/probability/blob/88d217dfe8be49050362eb14ba3076c0dc0f1ba6/tensorflow_probability/python/math/generic.py#L523-L574
    Args:
        x: data

    Returns:

    """
    threshold = np.log(ch.finfo(x.dtype).eps) + 2.
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = x.log()
    too_large_value = x
    device = ch.device("cuda:0" if x.is_cuda else 'cpu')
    x = ch.where(is_too_small | is_too_large, ch.ones([], dtype=x.dtype, device=device), x)
    y = x + ch.log(-ch.expm1(-x))  # == log(expm1(x))
    return ch.where(is_too_small, too_small_value, ch.where(is_too_large, too_large_value, y))
    # return (x.exp() - 1.).log()


def stable_cholesky(A, upper=False, out=None):
    chol, info = ch.linalg.cholesky_ex(A, upper=upper, out=out)
    if not ch.any(info):
        return chol

    # eps_float = 1e-6
    # eps_double = 1e-8
    eps = 1e-6

    diag_add = ((info > 0) * eps).unsqueeze(-1).expand(*A.shape[:-1])
    A.diagonal(dim1=-1, dim2=-2).add_(diag_add)
    logging.warning(f"A not PD, adding jitter of {eps:.1e} to the diagonal")
    chol, info = ch.linalg.cholesky_ex(A, upper=upper, out=out)
    if not ch.any(info):
        return chol
    else:
        raise ValueError(f"Matrix {A} not PD. Cholesky decomposition unsuccessful")

        # try:
    #     chol = ch.linalg.cholesky(A).transpose(-1, -2)
    # except RuntimeError:
    #     eigval, eigvec = ch.symeig(A, eigenvectors=True)
    #     if not ch.all(eigval >= -1e-8):
    #         raise ValueError(f"Covariance matrix {A} not PSD.")
    #     eigval_root = eigval.clamp_min(0.0).sqrt()
    #     chol = (eigvec * eigval_root).transpose(-1, -2)

    return chol
