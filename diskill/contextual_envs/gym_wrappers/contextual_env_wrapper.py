from typing import Union, Tuple

import gym
import numpy as np
from gym import Env


class ContextualEnvWrapper(gym.Wrapper):
    """
    A contextual environment wrapper. It includes a context space and a method to set the context of the environment.
    """
    def __init__(self, env: Env, **kwargs):
        super().__init__(env)

    context_space = None  # :gym_wrapper.spaces

    def seed(self, seed=None):
        self.context_space.seed(seed)
        super().seed(seed)

    def set_context(self, context: Union[Tuple, float, np.ndarray, int]):
        """
            Set the environments context. This externalizes the context selection and allows
            to include learning of its distribution

            :param context: Context to be applied onto the environment
        """
        assert self.context_space.contains(context), "context must be within bounds of the context space"

    def get_ctxt_punishment(self, context: Union[Tuple, float, np.ndarray, int],
                                  samples: Union[Tuple, float, np.ndarray, int]):
        """
            Returns a punishment for contexts outside of te context space. This allows
            to include learning of contexts through reward and punishment

            :param context: Context to calculate the punishment for
            :param samples: Corresponding samples sampled from the search distribution
        """
        raise NotImplementedError

    def sample_contexts(self, n_samples: int):
        """

        :param n_samples: number of context samples
        :return:
        """
        raise NotImplementedError

    def get_ctxt_dim(self):
        raise NotImplementedError

    def get_act_dim(self):
        raise NotImplementedError

    def get_ctxt_range(self):
        raise NotImplementedError

    def is_valid(self, ctxts):
        return np.array([self.context_space.contains(context) for context in ctxts])
