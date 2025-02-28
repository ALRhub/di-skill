from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch as ch


def concatenate(arr1, arr2, axis=0):
    try:
        return ch.cat((arr1, arr2), dim=axis)
    except Exception:
        return np.concatenate((arr1, arr2), axis=axis)


def valid_samples_from_buffer_subset(buffer):
    indices = buffer.valid_ctxt_indices
    all_data_dict = buffer.__dict__
    all_data = [all_data_dict[key][indices] for key in all_data_dict.keys()]
    return buffer.__class__(*all_data)


@dataclass
class BaseBufferSubset:
    ctxts: np.ndarray
    glob_ctxts: np.ndarray
    glob_samples: np.ndarray
    samples: np.ndarray
    rewards: np.ndarray
    sample_log_probs: np.ndarray
    ctxt_log_probs: np.ndarray

    def merge_buffer_subsets(self, other_buffer_subset):
        self.ctxts = concatenate(self.ctxts, other_buffer_subset.ctxts, axis=0)
        self.glob_ctxts = concatenate(self.glob_ctxts, other_buffer_subset.ctxts, axis=0)
        self.glob_samples = concatenate(self.glob_samples, other_buffer_subset.ctxts, axis=0)
        self.samples = concatenate(self.samples, other_buffer_subset.samples, axis=0)
        self.rewards = concatenate(self.rewards, other_buffer_subset.rewards, axis=0)
        self.sample_log_probs = concatenate(self.sample_log_probs, other_buffer_subset.sample_log_probs, axis=0)
        self.ctxt_log_probs = concatenate(self.ctxt_log_probs, other_buffer_subset.ctxt_log_probs, axis=0)

    def clean_based_iw(self, iw, use_iw):
        changed = False
        use_indices = None
        if use_iw:
            use_indices = np.where(iw > 1e-8)[0]
            changed = True if use_indices.shape[0] != iw.shape[0] else False
            self.ctxts = self.ctxts[use_indices]
            self.samples = self.samples[use_indices]
            self.rewards = self.rewards[use_indices]
            self.sample_log_probs = self.sample_log_probs[use_indices]
            self.ctxt_log_probs = self.ctxt_log_probs[use_indices]
        return changed, use_indices


@dataclass
class PGBufferSubset(BaseBufferSubset):
    old_values: Union[np.ndarray, ch.Tensor]
    advantages: Union[np.ndarray, ch.Tensor]
    advantages_mean: Union[np.ndarray, ch.Tensor]
    old_means: Union[np.ndarray, ch.Tensor]
    old_stds: Union[np.ndarray, ch.Tensor]
    sample_indices: Union[np.ndarray, ch.Tensor]

    def merge_buffer_subsets(self, other_buffer_subset):
        super(PGBufferSubset, self).merge_buffer_subsets(other_buffer_subset)
        self.old_values = concatenate(self.old_values, other_buffer_subset.old_values, axis=0)
        self.advantages = concatenate(self.advantages, other_buffer_subset.advantages, axis=0)
        self.advantages_mean = concatenate(self.advantages_mean, other_buffer_subset.advantages_mean, axis=0)
        self.old_means = concatenate(self.old_means, other_buffer_subset.old_means, axis=0)
        self.old_stds = concatenate(self.old_stds, other_buffer_subset.old_stds, axis=0)
        self.sample_indices = concatenate(self.sample_indices, other_buffer_subset.sample_indices, axis=0)

    def clean_based_iw(self, iw, use_iw):
        changed, use_indices = super(PGBufferSubset, self).clean_based_iw(iw, use_iw)
        if changed:
            self.old_values = self.old_values[use_indices]
            self.advantages = self.advantages[use_indices]
            self.advantages_mean = self.advantages_mean[use_indices]
            self.old_means = self.old_means[use_indices]
            self.old_stds = self.old_stds[use_indices]
            self.sample_indices = self.sample_indices[use_indices]
        return changed, use_indices
