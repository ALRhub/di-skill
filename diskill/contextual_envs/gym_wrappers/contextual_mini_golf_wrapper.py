from typing import Union, Tuple

import numpy as np
from fancy_gym.envs.mujoco.mini_golf.mini_golf_env import CONTEXT_BOUNDS, CONTEXT_BOUNDS_ONE_OBS

from gym.spaces import Box, flatdim

from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper


class ContextualMiniGolfEnvWrapper(ContextualEnvWrapper):
    """
    Contextual version of the TableTennis environment
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualMiniGolfEnvWrapper, self).__init__(env)
        self.min_context = CONTEXT_BOUNDS[0]
        self.max_context = CONTEXT_BOUNDS[1]
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def get_ctxt_dim(self):
        return self.context_space.shape[0]

    def get_act_dim(self):
        return self.env.action_space.shape[0]

    def sample_contexts(self, n_samples: int):
        ctxts = self.context_space.np_random.uniform(low=self.context_space.low, high=self.context_space.high,
                                                     size=(n_samples, self.context_space.shape[0]))
        return ctxts.astype(self.dtype)

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        super().set_context(context)
        return self.env.reset(options={'ctxt': context})

    def get_ctxt_range(self):
        return np.array([self.min_context, self.max_context])


class ContextualMiniGolfOneObsEnvWrapper(ContextualMiniGolfEnvWrapper):

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualMiniGolfOneObsEnvWrapper, self).__init__(env, dtype, **kwargs)
        self.min_context = CONTEXT_BOUNDS_ONE_OBS[0]
        self.max_context = CONTEXT_BOUNDS_ONE_OBS[1]
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)