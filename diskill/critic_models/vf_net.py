"""
Based on https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/models/value/vf_net.py.
Adjustions to the contextual, i.e. episodic case are made.
"""


from typing import Sequence

import torch as ch
import torch.nn as nn

from utils.network_utils import get_activation, get_mlp, initialize_weights


class VFNet(nn.Module):

    def __init__(self, obs_dim: int, output_dim: int = 1, init: str = "orthogonal",
                 hidden_sizes: Sequence[int] = (64, 64), activation: str = "tanh", shared=False, global_critic=False):
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

        super().__init__()

        self.contextual = obs_dim > 0
        self.indim = obs_dim
        self.shared = shared
        self.global_critic = global_critic
        if self.contextual:
            self.activation = get_activation(activation)
            self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True)
            prev = obs_dim if len(hidden_sizes) == 0 else hidden_sizes[-1]
            self.final = self._get_final(prev, output_dim, init)
        else:
            self.final = self._get_final_parameter()

    def _get_final(self, prev_size, output_dim, init):
        final = nn.Linear(prev_size, output_dim)
        initialize_weights(final, init, scale=1.0)
        return final

    def _get_final_parameter(self):
        return nn.Parameter(ch.normal(0, 0.01, (1,)))

    def forward(self, x, train=True):
        """
        Forward pass of the value network
        Args:
            x: States to compute the value estimate for.
        Returns:
            The value of the states x
        """

        self.train(train)

        if self.contextual:
            for affine in self._affine_layers:
                x = self.activation(affine(x))
            final = self.final(x).squeeze(-1)
        else:
            final = self.final.expand(x.shape[0])
        return final
