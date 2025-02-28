from typing import Any
from utils.sample_selector import BaseSampleSelector
from utils.replay_buffer import BaseReplayBuffer
from utils.buffer_sub_sets import BaseBufferSubset
from utils.env_sampler_base import SamplerBase
from utils.functions import maybe_np, maybe_torch, sample_from_cat, sample_multinomial

import numpy as np
import torch as ch


class EnvSampler(SamplerBase):
    """This call is only responsible for sampling from the environment and not also sampling from the replay buffer."""

    def __init__(self, env_id: str, num_envs: int, num_test_envs: int, seed: int, use_ch: bool, cpu: bool, dtype,
                 n_env_ctxt_samples: int, env_kwargs=None, cpu_cores=None):
        if env_kwargs is None:
            env_kwargs = {}
        super(EnvSampler, self).__init__(env_id, num_envs, num_test_envs, seed, env_kwargs, cpu_cores)
        self.use_ch = use_ch
        self.cpu = cpu
        self.dtype = dtype
        self.device = ch.device("cuda:0" if not self.cpu else "cpu")
        self.n_env_ctxt_samples = n_env_ctxt_samples

    def execute_on_env(self, contexts, samples):
        n_samples = contexts.shape[0]
        rewards = np.zeros((n_samples, 1))
        env_infos = np.full(shape=n_samples, fill_value={})
        rewards[:, 0], env_infos[:] = self(maybe_np(self.use_ch, contexts), maybe_np(self.use_ch, samples))
        return rewards, env_infos

    def get_global_ctxts(self):
        return maybe_torch(self.use_ch, self.cpu, self.dtype, self.env.sample_contexts(self.n_env_ctxt_samples))

    def get_samples(self, model: Any, n_samples: int, cmp_idx: int, glob_ctxts: ch.Tensor):
        glob_samples = model.components[cmp_idx].sample(glob_ctxts)
        with ch.no_grad():
            ctxt_cat_distr = ch.exp(model.ctxt_distribution[cmp_idx](glob_ctxts)).squeeze()
            indices = sample_from_cat(ctxt_cat_distr.squeeze(), n_samples, normalize=False)
            sampled_contexts = glob_ctxts[indices]
            samples = glob_samples[indices, :]
        rewards, env_infos = self.execute_on_env(sampled_contexts, samples)
        return rewards, env_infos, sampled_contexts, samples, indices, glob_samples

    def test_model(self, model: Any, n_samples: int, deterministic=True, max_gating=False):
        ctxts = self.test_env.sample_contexts(n_samples)
        with ch.no_grad():
            ctxts_mb_torch = maybe_torch(self.use_ch, self.cpu, self.dtype, ctxts)
            if deterministic:
                if max_gating:
                    gating_probs = model.gating_probs(ctxts_mb_torch)
                    cmp_idx = maybe_np(self.use_ch, ch.max(gating_probs, dim=1)[1])
                else:
                    _, cmp_idx = model.sample(ctxts_mb_torch)
                samples = model.get_means(ctxts_mb_torch, cmp_idx)
            else:
                samples, _ = model.sample(ctxts_mb_torch)
        samples = maybe_np(self.use_ch, samples)
        rewards, infos = self(ctxts, samples, training=False)  # TODO: infos is probably a list... convert to np.array
        return rewards, infos

    def test_model_mean_and_max(self, model: Any, n_samples: int):
        ctxts = self.test_env.sample_contexts(n_samples)
        with ch.no_grad():
            ctxts_mb_torch = maybe_torch(self.use_ch, self.cpu, self.dtype, ctxts)
            # mean eval
            _, cmp_idx = model.sample(ctxts_mb_torch)
            samples = model.get_means(ctxts_mb_torch, cmp_idx)
            samples = maybe_np(self.use_ch, samples)
            rewards_sampled, infos_sampled = self(ctxts, samples, training=False)

            # max cmp index
            gating_probs = model.gating_probs(ctxts_mb_torch)
            cmp_idx_max = maybe_np(self.use_ch, ch.max(gating_probs, dim=1)[1])
            samples_max = model.get_means(ctxts_mb_torch, cmp_idx_max)
            samples_max = maybe_np(self.use_ch, samples_max)
            rewards_max, infos_max = self(ctxts, samples_max, training=False)
        return [rewards_sampled, rewards_max], [infos_sampled, infos_max]

    def sample_contexts_only(self, n_samples: int):
        return maybe_torch(self.use_ch, self.cpu, self.dtype, self.test_env.sample_contexts(n_samples))


class SampleManager:
    def __init__(self, env_sampler: EnvSampler, sample_selector: BaseSampleSelector):
        self.env_sampler = env_sampler
        self.sample_selector = sample_selector

    def get_global_ctxts(self):
        return self.env_sampler.get_global_ctxts()

    def test_model(self, model: Any, n_test_samples: int, deterministic=True, max_gating=False):
        return self.env_sampler.test_model(model, n_test_samples, deterministic, max_gating=max_gating)

    def test_model_mean_and_max(self, model: Any, n_test_samples: int):
        return self.env_sampler.test_model_mean_and_max(model, n_test_samples)

    def _sample_env_one_cmp(self, model, n_samples_cmp: int, replay_buffer: BaseReplayBuffer, idx: int, glob_ctxts):
        rewards, env_infos, contexts, samples, indices, glob_samples = self.env_sampler.get_samples(model,
                                                                                                    n_samples_cmp, idx,
                                                                                                    glob_ctxts)
        replay_buffer.add(contexts, samples, rewards, model, idx, glob_ctxts, sample_indices=indices,
                          vf=model.components[idx].critic, glob_samples=glob_samples)
        if 'executed' in env_infos[0]:
            n_ex_samples = 0
            for elem in env_infos:
                n_ex_samples += 1 if elem['executed'][-1] else 0

        else:
            n_ex_samples = rewards.shape[0]
        n_ex_time_steps = 0
        for elem in env_infos:
            n_ex_time_steps += elem.pop('trajectory_length', 0)
        return n_ex_samples, n_ex_time_steps

    def sample_from_env(self, model: Any, n_samples_cmp: int, replay_buffer: BaseReplayBuffer, glob_ctxts, idx=None):
        with ch.no_grad():
            idx = np.ones(model.num_components) if idx is None else idx
            n_cmps = np.where(idx == 1)[0].shape[0]
            ctxts_smpls = ch.zeros(size=(n_cmps * n_samples_cmp, model.ctxt_dim), dtype=self.env_sampler.dtype,
                                   device=self.env_sampler.device)
            param_smpls = ch.zeros(size=(n_cmps * n_samples_cmp, model.smpl_dim), dtype=self.env_sampler.dtype,
                                   device=self.env_sampler.device)
            option_indices = ch.zeros(size=(n_cmps * n_samples_cmp, 1), dtype=self.env_sampler.dtype,
                                      device=self.env_sampler.device)
            glob_smpls = ch.zeros(size=(n_cmps * glob_ctxts.shape[0], model.smpl_dim), dtype=self.env_sampler.dtype,
                                  device=self.env_sampler.device)
            n_active_cmps = 0
            for cmp_idx in range(model.num_components):
                if idx[cmp_idx]:
                    cmp_glob_smpls = model.components[cmp_idx].sample(glob_ctxts)
                    ctxt_cat_distr = ch.exp(model.ctxt_distribution[cmp_idx](glob_ctxts))
                    indices = sample_from_cat(ctxt_cat_distr.squeeze(), n_samples_cmp, normalize=False)
                    ctxts_smpls[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp] = glob_ctxts[indices]
                    param_smpls[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp] = cmp_glob_smpls[
                        indices]
                    option_indices[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp, 0] = indices
                    n_active_cmps += 1

            # now execute all on env and then correctly put data in replay buffer
            rewards, env_infos = self.env_sampler.execute_on_env(ctxts_smpls, param_smpls)
            n_active_cmps = 0
            for cmp_idx in range(model.num_components):
                if idx[cmp_idx]:
                    replay_buffer.add(ctxts_smpls[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp],
                                      param_smpls[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp],
                                      rewards[n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp], model,
                                      cmp_idx, glob_ctxts,
                                      sample_indices=option_indices[
                                                     n_active_cmps * n_samples_cmp:(n_active_cmps + 1) * n_samples_cmp],
                                      vf=model.components[cmp_idx].critic,
                                      glob_samples=glob_smpls[n_active_cmps * glob_ctxts.shape[0]:
                                                              (n_active_cmps + 1) * glob_ctxts.shape[0]])
                    n_active_cmps += 1
            if 'executed' in env_infos[0]:
                n_ex_samples = 0
                for elem in env_infos:
                    n_ex_samples += 1 if elem['executed'][-1] else 0

            else:
                n_ex_samples = rewards.shape[0]
            n_ex_time_steps = 0
            for elem in env_infos:
                n_ex_time_steps += elem.pop('trajectory_length', 0)
            return n_ex_samples, n_ex_time_steps

    def get_samples(self, model: Any, cmp_idx: int, batch_size: int, replay_buffer: BaseReplayBuffer) \
            -> BaseBufferSubset:
        samples = self.sample_selector.select_samples(replay_buffer, batch_size, idx=cmp_idx, model=model)
        return samples

    def sample_contexts_only(self, n_samples: int):
        return self.env_sampler.sample_contexts_only(n_samples)
