from typing import Union, Tuple

import numpy as np
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import rot_to_quat
from gym.spaces import Box, flatdim

from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BOX_POS_BOUND
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_max, q_min
from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper
from utils.env_util import create_envs


class ContextualBoxPushingEnvWrapper(ContextualEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualBoxPushingEnvWrapper, self).__init__(env)
        self.min_context = BOX_POS_BOUND[0]
        self.min_context = np.concatenate((self.min_context, [0]))  # 0 for angle of orientation of target box
        self.max_context = BOX_POS_BOUND[1]
        self.max_context = np.concatenate(
            (self.max_context, [2 * np.pi]))  # 2*pi for angle of orientation of target box
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def get_ctxt_dim(self):
        return self.context_space.shape[0]

    def get_act_dim(self):
        return self.env.action_space.shape[0]

    def sample_contexts(self, n_samples: int):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                               self.context_space.shape[
                                                                                                   0]))
        done = False
        while not done:
            diff = np.linalg.norm(ctxts[:, :2] - self.env.box_init_pos[:2][None, :], axis=1)
            inv_ctxts_indices = np.where(diff < 0.3)[0]
            if inv_ctxts_indices.shape[0] != 0:
                ctxts[inv_ctxts_indices] = self.context_space.np_random.uniform(self.min_context, self.max_context,
                                                                                size=(inv_ctxts_indices.shape[0],
                                                                                      self.context_space.shape[0]))
            else:
                done = True
        return ctxts.astype(self.dtype)

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        super().set_context(context)
        return self.env.reset(options={'ctxt': context})

    def get_ctxt_range(self):
        return np.array([self.min_context, self.max_context])


class ContextualBoxPushingRotInvEnvWrapper(ContextualBoxPushingEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualBoxPushingRotInvEnvWrapper, self).__init__(env)
        self.min_context = BOX_POS_BOUND[0]
        self.min_context = np.concatenate((self.min_context, [0]))  # 0 for angle of orientation of target box
        self.max_context = BOX_POS_BOUND[1]
        self.max_context = np.concatenate((self.max_context, [89. * (np.pi / 180)]))
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)


class ContextualBoxPushingObstacleEnvWrapper(ContextualBoxPushingEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(ContextualBoxPushingObstacleEnvWrapper, self).__init__(env)
        from fancy_gym.envs.mujoco.box_pushing.box_pushing_obstacle_env import OBSTACLE_POS_BOUND, BOX_POS_BOUND
        self.min_context = np.concatenate((BOX_POS_BOUND[0], [0], OBSTACLE_POS_BOUND[0]))
        self.max_context = np.concatenate((BOX_POS_BOUND[1], [2 * np.pi], OBSTACLE_POS_BOUND[1]))
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def sample_contexts(self, n_samples: int):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                               self.context_space.shape[
                                                                                                   0]))

        done = False
        while not done:
            cond_1 = np.zeros(n_samples)
            cond_2 = np.zeros(n_samples)
            cond_3 = np.zeros(n_samples)
            cond_4 = np.zeros(n_samples)
            cond_5 = np.zeros(n_samples)

            target_box_pos = ctxts[:, :2]
            obs_pos = ctxts[:, -2:]
            target_init_diff = np.linalg.norm(target_box_pos - self.env.box_init_pos[:2][None, :], axis=1)
            cond_1[target_init_diff > 0.3] = 1
            cond_2[obs_pos[:, 1] < self.env.box_init_pos[1]] = 1
            cond_3[np.abs(obs_pos[:, 1] - self.env.box_init_pos[1]) >= 0.15] = 1
            cond_4[obs_pos[:, 1] > target_box_pos[:, 1]] = 1
            cond_5[np.abs(obs_pos[:, 1] - target_box_pos[:, 1]) >= 0.15] = 1
            all_conditions = cond_1 * cond_2 * cond_3 * cond_4 * cond_5
            invalid = np.where(all_conditions == 0)[0]
            if invalid.shape[0] != 0:
                ctxts[invalid] = self.context_space.np_random.uniform(self.min_context, self.max_context,
                                                                      size=(invalid.shape[0],
                                                                            self.context_space.shape[0]))
            else:
                done = True
        return ctxts.astype(self.dtype)


class ContextualBoxPushingRndm2Rndm(ContextualBoxPushingEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, ctxt_sample_env=False, buffer_size=50000, **kwargs):
        super(ContextualBoxPushingRndm2Rndm, self).__init__(env)
        self.min_context = BOX_POS_BOUND[0]  # min box pos of initial box
        self.min_context = np.concatenate((self.min_context, [0]))  # 0 for angle of orientation of initial box
        self.min_context = np.concatenate((self.min_context, q_min))  # minimum joint pos angles
        self.min_context = np.concatenate((self.min_context, BOX_POS_BOUND[0]))  # min box pos of target
        self.min_context = np.concatenate((self.min_context, [0]))  # min box orientation of target

        self.max_context = BOX_POS_BOUND[1]  # max box pos of initial box
        self.max_context = np.concatenate((self.max_context, [2 * np.pi]))  # max angle of orientation of initial box
        self.max_context = np.concatenate((self.max_context, q_max))  # max joint pos angles
        self.max_context = np.concatenate((self.max_context, BOX_POS_BOUND[1]))  # max box pos of target
        self.max_context = np.concatenate((self.max_context, [2 * np.pi]))  # max box pos of target

        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

        self.num_ctxt_sampler_envs = 20
        if ctxt_sample_env:
            self.ctxt_samplers = create_envs('VSL_ContextSamplerBoxPushingRndm2RndmProDMP-v0', 0,
                                             ## will be seeded externally
                                             self.num_ctxt_sampler_envs)
            self.init_ctxt_buffer = False
            self._buffer_size = buffer_size if ctxt_sample_env else 1
            self.contexts_buffer = self.sample_contexts(self._buffer_size)
            self._n_sampled_from_buffer = 0
        else:
            self.ctxt_samplers = None

    def seed(self, seed=None):
        super().seed(seed)
        if self.ctxt_samplers is not None:
            seed_sampler = seed + 14357 if seed is not None else None
            self.ctxt_samplers.seed(
                seed_sampler)  # Asyncvec env takes care of seeding each env individually with +1 value

    def _sample_valid_box_positions(self, n_samples):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                               self.context_space.shape[
                                                                                                   0]))
        done = False
        while not done:
            diff = np.linalg.norm(ctxts[:, -3:-1] - ctxts[:, :2], axis=1)
            inv_ctxts_indices = np.where(diff < 0.3)[0]
            if inv_ctxts_indices.shape[0] != 0:
                ctxts[inv_ctxts_indices] = self.context_space.np_random.uniform(self.min_context, self.max_context,
                                                                                size=(inv_ctxts_indices.shape[0],
                                                                                      self.context_space.shape[0]))
            else:
                done = True
        return ctxts

    def sample_contexts(self, n_samples: int):
        if self.init_ctxt_buffer:
            if self._n_sampled_from_buffer >= self._buffer_size * 20:
                self.init_ctxt_buffer = False
                self._n_sampled_from_buffer = 0
            else:
                self._n_sampled_from_buffer += n_samples
            idxs = self.context_space.np_random.randint(0, self.contexts_buffer.shape[0] - 1, size=n_samples)
            ctxts = self.contexts_buffer[idxs]
        else:
            ctxts = self.sample_new_contexts(self._buffer_size)
            self.init_ctxt_buffer = True
        return ctxts

    def sample_new_contexts(self, n_samples: int):
        n_ctxt_samples_per_env = np.ceil(n_samples / self.num_ctxt_sampler_envs)
        n_ctxt_samples_list = np.ones(self.num_ctxt_sampler_envs, dtype=np.int64) * int(n_ctxt_samples_per_env)
        ctxts = self.ctxt_samplers.sample_contexts(n_ctxt_samples_list)
        return np.vstack(ctxts)[:n_samples]


class ContextualBoxPushingRndm2RndmContextSampler(ContextualBoxPushingEnvWrapper):
    def __init__(self, env, dtype=np.float64, **kwargs):
        super().__init__(env, dtype, **kwargs)
        self.min_context = BOX_POS_BOUND[0]  # min box pos of initial box
        self.min_context = np.concatenate((self.min_context, [0]))  # 0 for angle of orientation of initial box
        self.min_context = np.concatenate((self.min_context, q_min))  # minimum joint pos angles
        self.min_context = np.concatenate((self.min_context, BOX_POS_BOUND[0]))  # min box pos of target
        self.min_context = np.concatenate((self.min_context, [0]))  # min box orientation of target

        self.max_context = BOX_POS_BOUND[1]  # max box pos of initial box
        self.max_context = np.concatenate((self.max_context, [2 * np.pi]))  # max angle of orientation of initial box
        self.max_context = np.concatenate((self.max_context, q_max))  # max joint pos angles
        self.max_context = np.concatenate((self.max_context, BOX_POS_BOUND[1]))  # max box pos of target
        self.max_context = np.concatenate((self.max_context, [2 * np.pi]))  # max box pos of target

        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)

    def _sample_valid_box_positions(self, n_samples):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                               self.context_space.shape[
                                                                                                   0]))
        done = False
        while not done:
            diff = np.linalg.norm(ctxts[:, -3:-1] - ctxts[:, :2], axis=1)
            inv_ctxts_indices = np.where(diff < 0.3)[0]
            if inv_ctxts_indices.shape[0] != 0:
                ctxts[inv_ctxts_indices] = self.context_space.np_random.uniform(self.min_context, self.max_context,
                                                                                size=(inv_ctxts_indices.shape[0],
                                                                                      self.context_space.shape[0]))
            else:
                done = True
        return ctxts

    def sample_contexts(self, n_samples: int):
        ctxts = self._sample_valid_box_positions(n_samples)
        ctxts[:, 3:10] = self.env.get_init_joint_pos(ctxts[:, :2])
        return ctxts
