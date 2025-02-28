from typing import Union, Tuple

import numpy as np
from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import check_init_states_valid_function, \
    is_init_state_valid_only_rndm_pos_batch, is_init_state_valid_batch
from gym.spaces import Box, flatdim

from fancy_gym.envs.mujoco.table_tennis.table_tennis_env import CONTEXT_BOUNDS_4DIMS, CONTEXT_BOUNDS_5DIMS

from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper


class ContextualTableTennisEnvWrapper(ContextualEnvWrapper):
    """
    Contextual version of the TableTennis environment
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualTableTennisEnvWrapper, self).__init__(env)
        self.min_context = CONTEXT_BOUNDS_4DIMS[0]
        self.max_context = CONTEXT_BOUNDS_4DIMS[1]
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def get_ctxt_dim(self):
        return self.context_space.shape[0]

    def get_act_dim(self):
        return self.env.action_space.shape[0]

    def sample_contexts(self, n_samples: int):
        init_ball_states = self._generate_valid_init_ball(n_samples)
        goal_positions = self.context_space.np_random.uniform(low=self.min_context[-2:], high=self.max_context[-2:],
                                                              size=(n_samples, 2))
        ctxts = np.concatenate((init_ball_states, goal_positions), axis=1)
        return ctxts.astype(self.dtype)

    def _generate_random_balls(self, n_samples):
        x_pos = self.context_space.np_random.uniform(low=self.min_context[0], high=self.max_context[0],
                                                     size=(n_samples, 1))
        y_pos = self.context_space.np_random.uniform(low=self.min_context[1], high=self.max_context[1],
                                                     size=(n_samples, 1))
        # init_ball_state = np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])
        init_ball_state = np.concatenate((x_pos, y_pos), axis=1)
        return init_ball_state

    def _generate_valid_init_ball(self, n_samples):
        init_ball_states = self._generate_random_balls(n_samples)
        done = False
        while not done:
            invalid_indices = np.where(is_init_state_valid_only_rndm_pos_batch(init_ball_states) == 0)[0]
            n_invalid_indices = invalid_indices.shape[0]
            if n_invalid_indices != 0:
                init_ball_states[invalid_indices] = self._generate_random_balls(n_invalid_indices)
            else:
                done = True
        return init_ball_states

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        super().set_context(context)
        return self.env.reset(options={'ctxt': context})

    def get_ctxt_range(self):
        return np.array([self.min_context, self.max_context])


class ContextualTableTennisVelEnvWrapper(ContextualTableTennisEnvWrapper):
    def __init__(self, env, dtype=np.float64, **kwargs):
        super().__init__(env, dtype=dtype, **kwargs)
        self.min_context = CONTEXT_BOUNDS_5DIMS[0]
        self.max_context = CONTEXT_BOUNDS_5DIMS[1]
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def sample_contexts(self, n_samples: int):
        init_ball_states = self._generate_valid_init_ball(n_samples)
        goal_positions = self.context_space.np_random.uniform(low=self.min_context[-2:], high=self.max_context[-2:],
                                                              size=(n_samples, 2))
        ctxts = np.concatenate((init_ball_states, goal_positions), axis=1)
        return ctxts.astype(self.dtype)

    def _generate_random_balls(self, n_samples):
        x_pos = self.context_space.np_random.uniform(low=self.min_context[0], high=self.max_context[0],
                                                     size=(n_samples, 1))
        y_pos = self.context_space.np_random.uniform(low=self.min_context[1], high=self.max_context[1],
                                                     size=(n_samples, 1))
        x_vels = self.context_space.np_random.uniform(low=self.min_context[2], high=self.max_context[2],
                                                      size=(n_samples, 1))
        # init_ball_state = np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])
        init_ball_state = np.concatenate((x_pos, y_pos, x_vels), axis=1)
        return init_ball_state

    def _generate_valid_init_ball(self, n_samples):
        init_ball_states = self._generate_random_balls(n_samples)
        done = False
        while not done:
            invalid_indices = np.where(is_init_state_valid_batch(init_ball_states) == 0)[0]
            n_invalid_indices = invalid_indices.shape[0]
            if n_invalid_indices != 0:
                init_ball_states[invalid_indices] = self._generate_random_balls(n_invalid_indices)
            else:
                done = True
        return init_ball_states
