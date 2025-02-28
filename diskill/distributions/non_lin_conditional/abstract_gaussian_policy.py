import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

import torch as ch
import torch.nn as nn
from critic_models.vf_net import VFNet

from utils.network_utils import get_activation, get_mlp, initialize_weights
from utils.torch_utils import tensorize, inverse_softplus
"""
Based on https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/models/policy/abstract_gaussian_policy.py.
Adjustions to the contextual, i.e. episodic case are made. Additional methods are written to provide usage for the 
Di-SkilL interface.
"""

class AbstractGaussianPolicy(nn.Module, ABC):

    def __init__(self, obs_dim: int, action_dim: Union[int, Tuple], init: str = "normal",
                 hidden_sizes: Sequence[int] = (64, 64), activation: str = "tanh", contextual_std: bool = False,
                 init_std: float = 1., init_mean: float = 0., minimal_std: float = 1e-5, device: ch.device = "cpu"):
        """
        Abstract Method defining a Gaussian policy structure.
        Args:
            obs_dim: Observation dimensionality aka input dimensionality
            action_dim: Action dimensionality aka output dimensionality
            init: Initialization type for the layers
            hidden_sizes: Sequence of hidden layer sizes for each hidden layer in the neural network.
            activation: Type of activation for hidden layers
            contextual_std: Whether to use a contextual standard deviation or not
            init_std: initial value of the standard deviation matrix
            init_mean: initial value of the mean vector
            minimal_std: minimal standard deviation

        Returns:

        """
        super().__init__()

        self.action_dim = action_dim
        self.cpu = True if device.type == 'cpu' else False
        self.minimal_std = tensorize(minimal_std, cpu=self.cpu, dtype=ch.float64)
        self.init_std = tensorize(init_std, cpu=self.cpu, dtype=ch.float64)

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        self.contextual = obs_dim > 0
        self.contextual_std = self.contextual and contextual_std
        prev_size = hidden_sizes[-1] if len(hidden_sizes) > 0 else obs_dim

        if self.contextual:
            self.activation = get_activation(activation)
            self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True, activation=activation)

        # This shift is applied to the Parameter/cov NN output before applying the transformation
        # and gives hence the wanted initial cov

        self._pre_activation_shift = self._get_preactivation_shift(self.init_std, self.minimal_std)
        self._mean = self._get_mean(self.contextual, action_dim, prev_size, init, init_mean=init_mean)
        self._pre_std = self._get_std(contextual_std and self.contextual, action_dim, prev_size, init)
        self.critic = None

    def set_critic(self, critic: VFNet):
        self.critic = critic

    @abstractmethod
    def forward(self, x, train=True):
        raise NotImplementedError()

    def _get_mean(self, contextual, action_dim, prev_size=None, init=None, scale=0.01, init_mean: float = 1e-3):
        """
        Constructor method for mean prediction. Do not overwrite.
        Args:
            contextual: whether to make the mean context dependent or not
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.

        Returns:
            Standard deviation parametrization.
        """
        # Flatten action space for network prediction
        action_dim = np.prod(action_dim).item()
        if contextual:
            return self._get_mean_layer(action_dim, prev_size, init, scale=scale, init_mean=init_mean)
        else:
            return self._get_mean_parameter(action_dim, scale=scale, init_mean=init_mean)

    def _get_mean_layer(self, action_dim, prev_size=None, init=None, scale=0.01, init_mean: float = 1e-3):
        """
        Constructor method for mean prediction.
        Args:
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.
            scale

        Returns:
            Mean parametrization.
        """
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, init, init_w=init_mean, scale=scale)
        return mean

    def _get_mean_parameter(self, action_dim, scale=0.01, init_mean=1e-3):
        """
        Creates a trainable variable for predicting the mean for a non contextual policy.
        Args:
            action_dim: Action dimension for output shape

        Returns:
            Torch trainable variable for mean prediction.
        """
        if isinstance(init_mean, Sequence):
            init_mean = ch.Tensor(init_mean)
            init = ch.normal(init_mean, scale)
        else:
            init = ch.normal(init_mean, scale, (action_dim,))
        return nn.Parameter(init)

    # @final
    def _get_std(self, contextual_std: bool, action_dim, prev_size=None, init=None):
        """
        Constructor method for mean prediction. Do not overwrite.
        Args:
            contextual_std: whether to make the mean context dependent or not
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.

        Returns:
            Standard deviation parametrization.
        """
        if contextual_std:
            return self._get_std_layer(prev_size, action_dim, init)
        else:
            return self._get_std_parameter(action_dim)

    def _get_preactivation_shift(self, init_std, minimal_std):
        """
        Compute the prediction shift to enforce an initial covariance value for contextual and non contextual policies.
        Args:
            init_std: value to initialize the covariance output with.
            minimal_std: lower bound on the covariance.

        Returns:
            Preactivation shift to enforce minimal and initial covariance.
        """
        return self.diag_activation_inv(init_std - minimal_std)

    @abstractmethod
    def _get_std_parameter(self, action_dim):
        """
        Creates a trainable variable for predicting the mean for a non contextual policy.
        Args:
            action_dim: Action dimension for output shape

        Returns:
            Torch trainable variable for covariance prediction.
        """
        pass

    @abstractmethod
    def _get_std_layer(self, prev_size, action_dim, init):
        """
        Creates a layer for predicting the mean for a contextual policy.
        Args:
            prev_size: Previous layer's output size
            action_dim: Action dimension for output shape
            init: Initialization type of layer.

        Returns:
            Torch layer for covariance prediction.
        """
        pass

    def sample(self, observation, n=1) -> ch.Tensor:
        p = self(observation)
        return self._sample(p, n)

    def log_density(self, observation: ch.Tensor, x: ch.Tensor, **kwargs):
        p = self(observation)
        return self._log_density(p, x, **kwargs)

    def log_density_fixed(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor):
        return self._log_density(p, x)

    def expected_entropy(self, observation: ch.Tensor):
        p = self(observation)
        return self.entropy(p).detach()

    @abstractmethod
    def _sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITHOUT reparametrization trick
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples

        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITH reparametrization trick.
        This version applies the reparametrization trick to allow for backpropagate through it.
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples
        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def _log_density(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs) -> ch.Tensor:
        """
        Computes the log probability of x given a batched distributions p (mean, mean)
        Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
            x: Values to compute logpacs for
            **kwargs:

        Returns:
            Log probabilities of x.
        """
        pass

    @abstractmethod
    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]) -> ch.Tensor:
        """
        Get entropies over the probability distributions given by p = (mean, var).
        mean shape (batch_size, action_space), var shape (action_space,)
        Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).

        Returns:
            Policy entropy based on sampled distributions p.
        """
        pass

    @abstractmethod
    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        """
        Returns the log determinant of the mean matrix
        Args:
            std: either a diagonal, cholesky, or sqrt matrix depending on the policy
        Returns:
            The log determinant of mean, aka log sum the diagonal
        """
        pass

    @abstractmethod
    def maha(self, mean, mean_other, std) -> ch.Tensor:
        """
        Compute the mahalanbis distance between two means. mean is the scaling matrix.
        Args:
            mean: left mean
            mean_other: right mean
            std: scaling matrix

        Returns:
            Mahalanobis distance between mean and mean_other
        """
        pass

    @abstractmethod
    def precision(self, std: ch.Tensor) -> ch.Tensor:
        """
        Compute precision matrix given the mean.
        Args:
            std: mean matrix

        Returns:
            Precision matrix
        """
        pass

    @abstractmethod
    def covariance(self, std) -> ch.Tensor:
        """
        Compute the full covariance matrix given the mean.
        Args:
            std: mean matrix

        Returns:

        """
        pass

    @abstractmethod
    def set_std(self, std: ch.Tensor) -> None:
        """
        For the NON-contextual standard deviation case we do not need to regress the standard deviation, we can simply set it.
        This is a helper method to achieve this.
        Args:
            std: projected mean

        Returns:

        """
        pass

    def set_std_weight(self, weight: ch.Tensor) -> None:
        """
        For the NON-contextual standard deviation case we do not need to regress the standard deviation weight, we can simply set it.
        This is a helper method to achieve this.
        Args:
            weight: projected scalar weight value of std

        Returns:

        """
        pass

    def set_mean(self, mean: ch.Tensor) -> None:
        """
        For the NON-contextual case we do not need to regress the mean, we can simply set it.
        This is a helper method to achieve this.
        Args:
            mean: projected std

        Returns:

        """
        assert not self.contextual
        self._mean.data = mean

    def get_last_layer(self):
        """
        Returns last layer of network. Only required for the PAPI projection.

        Returns:
            Last layer weights for PAPI prpojection.

        """
        return self._affine_layers[-1].weight.data

    def papi_weight_update(self, eta: ch.Tensor, A: ch.Tensor):
        """
        Update the last layer alpha according to papi paper [Akrour et al., 2019]
        Args:
            eta: Multiplier alpha from [Akrour et al., 2019]
            A: Projected intermediate policy matrix

        Returns:

        """
        self._affine_layers[-1].weight.data *= eta
        self._affine_layers[-1].weight.data += (1 - eta) * A

    @property
    def is_root(self):
        """
        Whether policy is returning a full sqrt matrix as mean.
        Returns:

        """
        return False

    @property
    def is_diag(self):
        """
        Whether the policy is returning a diagonal matrix as mean.
        Returns:

        """
        return False

    @property
    def is_scaled(self):
        return False
