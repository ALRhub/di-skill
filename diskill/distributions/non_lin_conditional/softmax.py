import torch as ch
import torch.nn as nn

from typing import Sequence

from utils.network_utils import get_mlp, get_activation, initialize_weights


class GatingNetwork(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 init_n_components: int,
                 max_n_components: int,
                 init: str = 'normal',
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = 'relu', device: ch.device = "cpu"):
        super(GatingNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.mask = ch.zeros([max_n_components], device=device)
        self.mask[:init_n_components] += 1
        self.n_components = init_n_components
        self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True, activation=activation)
        prev_size = hidden_sizes[-1] if len(hidden_sizes) > 0 else obs_dim
        self._last_layer = nn.Linear(prev_size, max_n_components)
        initialize_weights(self._last_layer, init, scale=0.01)
        self.activation = get_activation(activation)

    def forward(self, x, train=True):
        self.train(train)
        for affine in self._affine_layers:
            x = self.activation(affine(x))
        x = self._last_layer(x)
        x = self.mask * x
        return ch.nn.functional.log_softmax(x[:, :self.n_components], dim=1)

    def entropy(self, x):
        log_softmax = self.forward(x)
        probs = ch.exp(log_softmax)
        return -(probs * log_softmax).sum(dim=-1, keepdims=True)

    def expected_entropy(self, x):
        return self.entropy(x).mean()

    def entropy_specific_cmp(self, x, cmp_indices):
        log_softmax = self.forward(x)
        probs = ch.exp(log_softmax)
        all_entropies = -(probs * log_softmax).sum(dim=-1, keepdims=True)
        return all_entropies.gather(1, cmp_indices)

    def log_density_fixed(self, x, output_indices):
        x = self.forward(x)
        return x.gather(1, output_indices)

    def get_log_posterior(self, ctxts):
        # p(c) cancels out because we treat as categorical and uniformly distributed
        alpha = 1
        log_gating = (1 / alpha) * self(ctxts)
        normalizer = ch.logsumexp(log_gating, dim=0, keepdim=True)
        log_posterior = log_gating - normalizer
        return log_posterior.detach(), normalizer.detach()

    def marginal_prob(self, x):
        log_softmax = self.forward(x)
        probs = ch.exp(log_softmax)
        marginal_probs = probs.sum(dim=0)
        marginal_probs /= marginal_probs.sum()
        return marginal_probs

    def marginal_entropy(self, x):
        marginal_probs = self.marginal_prob(x)
        marginal_log_probs = ch.log(marginal_probs + 1e-20)
        return -(marginal_probs * marginal_log_probs).sum(dim=-1)

    @property
    def param_norm(self):
        """
        Calculates the norm of network parameters.
        """
        return ch.norm(ch.stack([ch.norm(p.detach()) for p in self.parameters()]))

    @property
    def grad_norm(self):
        """
        Calculates the norm of current gradients.
        """
        return ch.norm(ch.stack([ch.norm(p.grad.detach()) for p in self.parameters()]))

    def add_component(self):
        self.mask[self.n_components] = 1
        self.n_components += 1

    def remove_component(self, idx):
        self.mask[idx] = 0
        self.n_components -= 1


class ContextDistrNetwork(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 init: str = 'normal',
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = 'relu', device: ch.device = "cpu"):
        super(ContextDistrNetwork, self).__init__()
        self.obs_dim = obs_dim
        self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True, activation=activation)
        prev_size = hidden_sizes[-1] if len(hidden_sizes) > 0 else obs_dim
        self._last_layer = nn.Linear(prev_size, 1, bias=True)
        initialize_weights(self._last_layer, init, scale=0.01)
        self.activation = get_activation(activation)

    def forward(self, x, train=True):
        self.train(train)
        for affine in self._affine_layers:
            x = self.activation(affine(x))
        x = self._last_layer(x)
        return ch.nn.functional.log_softmax(x, dim=0)

    def log_density(self, x):
        return self(x)

    def entropy(self, x):
        log_softmax = self.forward(x)
        probs = ch.exp(log_softmax)
        return -(probs * log_softmax).sum(dim=0, keepdims=True)

    def add_component(self):
        raise NotImplementedError

    def remove_component(self, idx):
        raise NotImplementedError

    @property
    def num_components(self):
        return self.n_components
