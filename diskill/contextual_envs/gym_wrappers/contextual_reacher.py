from typing import Union, Tuple

import numpy as np
from gym.spaces import Box, flatdim
from contextual_envs.gym_wrappers.contextual_env_wrapper import ContextualEnvWrapper


# TODO: GENERALIZE THE CONTEXT SPACE

class Contextual5LinkReacherEnvWrapper(ContextualEnvWrapper):
    """
    Contextual version of the BeerpongEnv using the cup position as context
    """

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(Contextual5LinkReacherEnvWrapper, self).__init__(env)
        self.min_context = np.array([-0.5, -0.5])
        # self.max_context = np.array([0.0, 0.5])
        self.max_context = np.array([0.5, 0.5])
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)
        
    def get_ctxt_dim(self):
        return self.context_space.shape[0]

    def get_act_dim(self):
        return self.env.action_space.shape[0]

    def sample_contexts(self, n_samples: int):
        ctxts = self.context_space.np_random.uniform(self.min_context, self.max_context, size=(n_samples,
                                                                                       self.context_space.shape[0]))
        done = False
        while not done:
            ctxts_norm = np.linalg.norm(ctxts, axis=1)
            inv_ctxts_indices = np.where(ctxts_norm > 0.5)[0]
            if inv_ctxts_indices.shape[0] != 0:
                ctxts[inv_ctxts_indices] = self.context_space.np_random.uniform(self.min_context, self.max_context,
                                                        size=(inv_ctxts_indices.shape[0], self.context_space.shape[0]))
            else:
                done = True
        return ctxts.astype(self.dtype)

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        super().set_context(context)
        self.env.set_context(context)

    def get_ctxt_punishment(self, context: Union[Tuple, float, np.ndarray, int],
                            samples: Union[Tuple, float, np.ndarray, int]):
        if context.shape == self.context_space.shape:
            dist = np.linalg.norm(context)
            # consider all quadrants
            return 0 if dist < 0.5 else -150 - 100 * self._get_pun_single(context)

        returns = []
        for c in range(context.shape[0]):
            if np.linalg.norm(context[c]) < 0.5 and context[c][1] > 0:
                returns.append(0)
            else:
                # returns.append(-150 - 10*self._get_pun_single(context[c, :]))
                returns.append(-150 - 100 * self._get_pun_single(context[c, :]))
        return np.array(returns)

    def _get_pun_single(self, context: Union[Tuple, float, np.ndarray, int]):
        dist_condition = np.linalg.norm(context) > 0.5
        if dist_condition:
            return dist_condition * (np.linalg.norm(context) - 0.5) ** 2
        else:
            return 0

    def get_ctxt_range(self):
        return np.array([self.min_context, self.max_context])


class Contextual5LinkLEFTReacherEnvWrapper(Contextual5LinkReacherEnvWrapper):

    def __init__(self, env, dtype=np.float64, **kwargs):
        super(Contextual5LinkLEFTReacherEnvWrapper, self).__init__(env)
        self.min_context = np.array([-0.5, -0.5])
        self.max_context = np.array([0.0, 0.5])
        self.dtype = dtype
        self.context_space = Box(low=self.min_context, high=self.max_context, dtype=dtype)