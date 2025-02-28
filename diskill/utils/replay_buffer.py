import numpy as np
import torch as ch
from typing import Any
from collections import deque

from utils.buffer_sub_sets import BaseBufferSubset, PGBufferSubset
from utils.torch_utils import tensorize


class BaseReplayBuffer:
    def __init__(self, buffer_size: int, c_dim: int, a_dim: int, use_ch=False, device=None, dtype=None):
        self.pos = 0
        self.full = False
        self.buffer_size = buffer_size
        self.c_dim = c_dim
        self.a_dim = a_dim
        self.use_ch = use_ch
        self.dtype = dtype
        self.device = device
        self.ctxts = self._create_array((buffer_size, c_dim), dtype=self.dtype)
        self.glob_ctxts = None
        self.glob_samples = None
        self.samples = self._create_array((buffer_size, a_dim), dtype=self.dtype)
        self.rewards = self._create_array((buffer_size, 1), dtype=self.dtype)
        self.sample_log_probs = self._create_array(buffer_size, dtype=self.dtype)
        self.ctxt_log_probs = self._create_array(buffer_size, dtype=self.dtype)
        self.n_last_added_samples = 0

    def update(self, idx):
        self.n_last_added_samples = 0

    def _create_array(self, shape, dtype):
        if self.use_ch:
            return ch.zeros(shape, dtype=dtype, device=self.device)
        else:
            return np.zeros(shape)

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def _add(self, buffer_subset: BaseBufferSubset):
        self.ctxts[self.pos] = buffer_subset.ctxts
        # self.global_ctxts[self.pos] = buffer_subset.global_ctxts
        self.samples[self.pos] = buffer_subset.samples
        self.rewards[self.pos] = buffer_subset.rewards
        self.sample_log_probs[self.pos] = buffer_subset.sample_log_probs
        self.ctxt_log_probs[self.pos] = buffer_subset.ctxt_log_probs

    def add(self, ctxts: np.ndarray, samples: np.ndarray, rewards: np.ndarray, model: Any, idx: int, glob_ctxts,
            **kwargs):
        sample_log_probs = model.components[idx].log_density(ctxts, samples)
        ctxt_log_probs = model.context_components[idx].log_density(ctxts)
        self.n_last_added_samples += ctxts.shape[0]
        for i in range(ctxts.shape[0]):
            self._add(BaseBufferSubset(ctxts[i, :], None, samples[i, :], rewards[i, :],
                                       sample_log_probs[i], ctxt_log_probs[i]))
            self.pos += 1
            if self.pos > self.buffer_size - 1:
                self.pos = 0
                self.full = True

    def add_cmps(self, n_cmps: int):
        raise NotImplementedError

    def create_sample_set(self, indices, **kwargs):
        return BaseBufferSubset(self.ctxts[indices], self.glob_ctxts, self.glob_samples, self.samples[indices],
                                self.rewards[indices], self.sample_log_probs[indices], self.ctxt_log_probs[indices])

    def get_last_added(self, n_last_samples: int):
        if self.pos >= n_last_samples - 1:
            indices = np.linspace(self.pos - n_last_samples, self.pos, endpoint=False, dtype=int, num=n_last_samples)
        elif self.pos < n_last_samples - 1 and self.full:
            batch_rear_segment = np.linspace(0, self.pos, endpoint=False, dtype=int, num=self.pos)
            indices = np.concatenate((np.linspace(self.buffer_size - n_last_samples + self.pos,
                                                  self.buffer_size, endpoint=False, dtype=int,
                                                  num=n_last_samples - self.pos), batch_rear_segment))
        else:
            raise ValueError(f"Can not extract {n_last_samples}. The replay buffer only has {self.pos} samples right "
                             f"now")
        return self.create_sample_set(indices)


class PerCmpPGModReplayBufferSingle(BaseReplayBuffer):
    def __init__(self, buffer_size: int, c_dim: int, a_dim: int, norm_buffer_size: int, dtype, cpu, use_ch=True):
        super(PerCmpPGModReplayBufferSingle, self).__init__(buffer_size, c_dim, a_dim, use_ch,
                                                            ch.device("cuda:0" if not cpu else "cpu"), dtype)
        self.dtype = dtype
        self.device = ch.device("cuda:0" if not cpu else "cpu")
        self.cpu = cpu
        self.use_ch = use_ch
        self.old_values = self._create_array((buffer_size, 1), dtype=self.dtype)
        self.advantages = self._create_array((buffer_size, 1), dtype=self.dtype)
        self.advantages_mean = self._create_array((buffer_size, 1), dtype=self.dtype)
        self.old_means = self._create_array((buffer_size, a_dim), dtype=self.dtype)
        self.old_stds = self._create_array((buffer_size, a_dim, a_dim), dtype=self.dtype)
        self.sample_indices = self._create_array(buffer_size, dtype=ch.int64)
        self.rwd_q = deque(maxlen=norm_buffer_size * buffer_size)  # This is for a history mean
        self.norm_buffer_size = norm_buffer_size

    def _add(self, buffer_subset: PGBufferSubset):
        super(PerCmpPGModReplayBufferSingle, self)._add(buffer_subset)
        self.old_values[self.pos] = buffer_subset.old_values
        self.advantages[self.pos] = buffer_subset.advantages
        self.advantages_mean[self.pos] = buffer_subset.advantages_mean
        self.old_means[self.pos] = buffer_subset.old_means
        self.old_stds[self.pos] = buffer_subset.old_stds
        self.sample_indices[self.pos] = buffer_subset.sample_indices

    def add(self, ctxts: np.ndarray, samples: np.ndarray, rewards: np.ndarray, model: Any, idx: int, glob_ctxts,
            sample_indices=None, vf=None, glob_samples=None, **kwargs):
        self.rwd_q.extend(rewards.squeeze(axis=-1))
        ctxts_torch = tensorize(ctxts, cpu=self.cpu, dtype=self.dtype)
        glob_ctxts_torch = tensorize(glob_ctxts, cpu=self.cpu, dtype=self.dtype)
        samples_torch = tensorize(samples, cpu=self.cpu, dtype=self.dtype)
        rewards_torch = tensorize(rewards, cpu=self.cpu, dtype=self.dtype)
        self.n_last_added_samples = ctxts.shape[0]
        with ch.no_grad():
            old_means, old_stds = model.components[idx](ctxts_torch)
            sample_log_probs = model.components[idx].log_density_fixed((old_means, old_stds), samples_torch)
            ctxt_log_probs = model.ctxt_distribution[idx](ctxts_torch)
            mean_baseline = ch.ones((ctxts.shape[0], 1), device=self.device) * np.mean(self.rwd_q)
            if vf:
                old_values = vf(ctxts_torch)
            else:
                old_values = mean_baseline
            old_values = old_values[:, None] if len(old_values.shape) == 1 else old_values
            mean_baseline = mean_baseline[:, None] if len(mean_baseline.shape) == 1 else mean_baseline
            advantages = rewards_torch - old_values
            advantages_mean = rewards_torch - mean_baseline

        for i in range(ctxts.shape[0]):
            self._add(PGBufferSubset(ctxts_torch[i, :], None, None, samples_torch[i, :],
                                     rewards_torch[i, :], sample_log_probs[i],
                                     ctxt_log_probs[i], old_values[i, :], advantages[i, :], advantages_mean[i, :],
                                     old_means[i, :], old_stds[i, :], sample_indices[i]))
            self.pos += 1
            if self.pos > self.buffer_size - 1:
                self.pos = 0
                self.full = True
        self.glob_ctxts = glob_ctxts_torch
        self.glob_samples = glob_samples

    def create_sample_set(self, indices, **kwargs):

        return PGBufferSubset(self.ctxts[indices], self.glob_ctxts, self.glob_samples, self.samples[indices],
                              self.rewards[indices], self.sample_log_probs[indices], self.ctxt_log_probs[indices],
                              self.old_values[indices], self.advantages[indices], self.advantages_mean[indices],
                              self.old_means[indices], self.old_stds[indices], self.sample_indices[indices])


class PerCmpPGModReplayBuffer(PerCmpPGModReplayBufferSingle):
    def __init__(self, buffer_size: int, c_dim: int, a_dim: int, n_cmps: int, norm_buffer_size: int, dtype, cpu,
                 use_ch):
        super(PerCmpPGModReplayBuffer, self).__init__(buffer_size, c_dim, a_dim, norm_buffer_size, dtype, cpu, use_ch)
        self.n_cmps = n_cmps
        self.replay_buffer_list = [PerCmpPGModReplayBufferSingle(buffer_size, c_dim, a_dim, norm_buffer_size, dtype,
                                                                 cpu) for _ in range(n_cmps)]

    def update(self, idx):
        self.replay_buffer_list[idx].update(idx)

    def size(self) -> int:
        return self.replay_buffer_list[0].size()

    def add(self, ctxts: np.ndarray, samples: np.ndarray, rewards: np.ndarray, model: Any, idx: int, glob_ctxts,
            sample_indices=None, pre_squashed_ctxts=None, vf=None, glob_samples=None, **kwargs):
        self.replay_buffer_list[idx].add(ctxts, samples, rewards, model, idx, glob_ctxts, sample_indices=sample_indices,
                                         glob_samples=glob_samples, vf=vf, **kwargs)

    def create_sample_set(self, indices, **kwargs):
        return self.replay_buffer_list[kwargs['idx']].create_sample_set(indices)

    def delete_cmp(self, idx: int):
        del self.replay_buffer_list[idx]
        return len(self.replay_buffer_list)

    def sub_buffer(self, idx: int):
        return self.replay_buffer_list[idx]

    def add_cmps(self, n_cmp_adds: int):
        for _ in range(n_cmp_adds):
            self.replay_buffer_list.append(PerCmpPGModReplayBufferSingle(self.buffer_size, self.c_dim, self.a_dim,
                                                                         self.norm_buffer_size, self.dtype, self.cpu,
                                                                         self.use_ch))
        return len(self.replay_buffer_list)
