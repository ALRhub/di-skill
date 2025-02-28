import numpy as np
import torch as ch
import torch.nn as nn
from typing import Tuple

from distributions.non_lin_conditional.abstract_gaussian_policy import AbstractGaussianPolicy
from utils.network_utils import initialize_weights
"""
Based on https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/models/policy/gaussian_policy_diag.py.
Adjustions to the contextual, i.e. episodic case are made. Additional methods are written to provide usage for the 
Di-SkilL interface.
"""

class GaussianPolicyDiag(AbstractGaussianPolicy):

    """
    A Gaussian policy using a fully connected neural network.
    The parameterizing tensor is a mean and diagonal mean vector.
    """

    def _get_std_parameter(self, action_dim):
        std = ch.normal(0, 0.01, (action_dim,))
        return nn.Parameter(std)

    def _get_std_layer(self, prev_size, action_dim, init):
        std = nn.Linear(prev_size, action_dim)
        initialize_weights(std, init, scale=0.01)
        return std

    def forward(self, x, train=True):
        self.train(train)

        if self.contextual:
            for affine in self._affine_layers:
                x = self.activation(affine(x))

        std = self._pre_std(x) if self.contextual_std else self._pre_std
        std = (self.diag_activation(std + self._pre_activation_shift) + self.minimal_std)
        std = std.diag_embed().expand(x.shape[0], -1, -1)

        mean = self._mean(x) if self.contextual else self._mean.expand(x.shape[0], -1)
        return mean, std

    def _sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        means, std = p
        std = std.diagonal(dim1=-2, dim2=-1)
        eps = ch.randn((n,) + means.shape).to(dtype=std.dtype, device=std.device)
        samples = means + eps * std
        return samples.squeeze(0)

    def _log_density(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        mean, std = p
        k = x.shape[-1]

        maha_part = self.maha(x, mean, std)
        const = np.log(2.0 * np.pi) * k
        logdet = self.log_determinant(std)

        nll = -0.5 * (maha_part + const + logdet)
        return nll

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]):
        _, std = p
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return .5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a diagonal matrix
        Args:
            std: a diagonal matrix
        Returns:
            The log determinant of mean, aka log sum the diagonal
        """
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def precision(self, std: ch.Tensor):
        return (1 / self.covariance(std).diagonal(dim1=-2, dim2=-1)).diag_embed()

    def covariance(self, std: ch.Tensor):
        return std.pow(2)

    def set_std(self, std: ch.Tensor) -> None:
        assert not self.contextual_std
        shifted_min = self.minimal_std + ch.finfo(std.dtype).eps
        std_min = std.diagonal().clamp(min=shifted_min) - self.minimal_std
        self._pre_std.data = self.diag_activation_inv(std_min) - self._pre_activation_shift

    @property
    def is_diag(self):
        return True
