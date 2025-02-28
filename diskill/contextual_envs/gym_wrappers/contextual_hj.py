from typing import Union, Tuple

import numpy as np
from fancy_gym.envs.mujoco.hopper_jump.hopper_jump import CONTEXT_BOUNDS
from gym.spaces import Box, flatdim
from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper


# TODO: GENERALIZE THE CONTEXT SPACE

class ContextualHopperJumpEnvWrapper(ContextualEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualHopperJumpEnvWrapper, self).__init__(env)
        self.min_context = CONTEXT_BOUNDS[0]
        self.max_context = CONTEXT_BOUNDS[1]
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)
        
    def get_ctxt_dim(self):
        return self.context_space.shape[0]

    def get_act_dim(self):
        return self.env.action_space.shape[0]

    def sample_contexts(self, n_samples: int):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                       self.context_space.shape[0]))
        return ctxts.astype(self.dtype)

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        super().set_context(context)
        self.env.set_context(context)

    def get_ctxt_range(self):
        return np.array([self.min_context, self.max_context])
