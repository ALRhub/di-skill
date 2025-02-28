from typing import Sequence

import torch as ch
import torch.nn as nn

from critic_models.vf_net import VFNet
from utils.network_utils import initialize_weights


class QFNet(VFNet):

    def __init__(self, obs_dim: int, output_dim: int = 1, init: str = "fanin", hidden_sizes: Sequence[int] = (256, 256),
                 activation: str = "relu"):
        """
        A value network using a fully connected neural network.
        Args:
            obs_dim: Observation dimensionality aka input dimensionality
            output_dim: Action dimensionality aka output dimensionality, generally this is 1
            init: Initialization of layers
            hidden_sizes: Sequence of hidden layer sizes for each hidden layer in the neural network.
            activation: Type of activation for hidden layers

        Returns:

        """
        super().__init__(obs_dim, output_dim, init, hidden_sizes, activation)

    def _get_final(self, prev_size, output_dim, init, gain=1.0, scale=1 / 3):
        final = nn.Linear(prev_size, output_dim)
        initialize_weights(final, "uniform", init_w=3e-3)
        return final

    def forward(self, x, train=True):
        flat_inputs = ch.cat(x, dim=-1)
        return super().forward(flat_inputs, train=train)
