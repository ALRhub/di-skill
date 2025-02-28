from abc import ABC
from collections import defaultdict

import numpy as np
import os
from fancy_gym import make

from contextual_envs.contextual_async_vector_env import ContextualAsyncVectorEnv
from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper
from utils.env_util import create_envs


class SamplerBase(ABC):
    def __init__(self, env_id: str, num_envs: int = 1, num_test_envs: int = 1, seed: int = 0, env_kwargs=None,
                 cpu_cores=None):
        if env_kwargs is None:
            env_kwargs = {}
        self._is_common_async_env = True
        self.cpu_cores = cpu_cores
        try:
            self.async_envs = create_envs(env_id, seed, num_envs, **env_kwargs)
            self.test_async_envs = create_envs(env_id, seed + num_envs, num_test_envs, **env_kwargs)
            ########
            if cpu_cores is not None and num_envs > 1:
                cores_per_env = int(len(self.cpu_cores) / num_envs)
                cores_per_test_env = int(len(self.cpu_cores) / num_test_envs)
                cpu_cores_list = list(self.cpu_cores)
                env_pids = [self.async_envs.processes[i].pid for i in range(num_envs)]
                test_env_pids = [self.test_async_envs.processes[i].pid for i in range(num_test_envs)]
                for i, pid in enumerate(env_pids):
                    cores_env = cpu_cores_list[i * cores_per_env: (i + 1) * cores_per_env]
                    os.sched_setaffinity(pid, set(cores_env))
                for j, test_pid in enumerate(test_env_pids):
                    test_cores_env = cpu_cores_list[j * cores_per_test_env: (j + 1) * cores_per_test_env]
                    os.sched_setaffinity(test_pid, set(test_cores_env))
            self._single_env = make(env_id, seed, **env_kwargs, ctxt_sample_env=True)
            self._single_test_env = make(env_id, seed + num_envs, **env_kwargs, ctxt_sample_env=True)
        except (ValueError, AttributeError):
            raise ValueError("Could not create environment. Is it defined?")
        self.num_envs = num_envs
        self.num_test_envs = num_test_envs
        self._n_environment_steps = 0
        self._n_online_evaluated_episodes = 0

    def __call__(self, contexts: np.ndarray, samples: np.ndarray, training=True):
        if training:
            env = self.async_envs
            num_envs = self.num_envs
        else:
            env = self.test_async_envs
            num_envs = self.num_test_envs
        # TODO: The if-else is a hack to have support for the PLReacher envs
        if self._is_common_async_env:
            samples = np.atleast_2d(samples)
            contexts = np.atleast_2d(contexts)
            if samples.shape[0] == 0:
                return [], {}
            n_samples = samples.shape[0]
            split_parameters = np.array_split(samples, np.ceil(n_samples / num_envs))
            split_contexts = np.array_split(contexts, np.ceil(n_samples / num_envs))

            vals = defaultdict(list)
            for p, c in zip(split_parameters, split_contexts):
                env.reset()
                env.set_context(c)
                obs, reward, done, info = env.step(p)
                vals['obs'].append(obs)
                vals['reward'].append(reward)
                vals['done'].append(done)
                vals['info'].append(info)
            returns = np.hstack(vals['reward'])[:n_samples]
            infos = self._flatten_list(vals['info'])[:n_samples]
        else:
            returns, infos = self.async_envs(contexts, samples)
            infos = [infos]
        self._n_online_evaluated_episodes += returns.shape[0]
        return returns, infos  # , self._n_online_evaluated_episodes

    @staticmethod
    def _flatten_list(l):
        assert isinstance(l, (list, tuple))
        assert len(l) > 0
        assert all([len(l_) > 0 for l_ in l])
        return [l__ for l_ in l for l__ in l_]

    def reset(self):
        pass

    @property
    def env(self) -> ContextualEnvWrapper:
        return self._single_env

    @property
    def test_env(self) -> ContextualEnvWrapper:
        return self._single_test_env

    @property
    def n_environment_steps(self) -> int:
        return self._n_environment_steps
