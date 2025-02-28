import numpy as np
from typing import Union
import torch as ch
from torch.distributions import Categorical

from utils.torch_utils import tensorize, get_numpy


def log_sum_exp(exp_arg, axis):
    exp_arg_use = exp_arg.copy()
    max_arg = np.max(exp_arg_use)
    exp_arg_use = np.clip(exp_arg_use - max_arg, -700, 700)
    return max_arg + np.log(np.sum(np.exp(exp_arg_use), axis=axis))


def maybe_torch(use_ch: bool, cpu: bool, dtype, x: Union[np.ndarray, ch.Tensor]):
    return tensorize(x, cpu, dtype=dtype) if use_ch else x


def maybe_np(use_ch: bool, x: Union[np.ndarray, ch.Tensor]):
    return get_numpy(x) if use_ch else x


def make_tensor(x: list, axis: int):
    try:
        c_x = x[0]
        for i in range(len(x) - 1):
            c_x = ch.concat((c_x, x[i + 1]), dim=axis)
    except Exception:
        c_x = np.concatenate(x, axis=axis)
    return c_x


def concatenate(x1: Union[np.ndarray, ch.Tensor], x2: [np.ndarray, ch.Tensor], axis: int):
    try:
        return ch.concat((x1, x2), dim=axis)
    except Exception:
        return np.concatenate((x1, x2), axis=axis)


def sample_from_cat(probs: Union[np.ndarray, ch.Tensor], n_samples: int, normalize=True):
    if normalize:
        return Categorical(logits=probs).sample((n_samples,))
    else:
        return Categorical(probs).sample((n_samples,))


def sample_multinomial(probs: Union[np.ndarray, ch.Tensor], replacement, n_samples: int):
    return ch.multinomial(probs, n_samples, replacement)