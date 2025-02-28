from typing import Sequence

import numpy as np
import torch as ch

from distributions.non_lin_conditional.policy_factory import get_policy_network
from utils.functions import log_sum_exp
from utils.torch_utils import get_numpy, tensorize


class CtxtDistrNonLinExpMOE:
    def __init__(self, ctxt_dim: int,
                 sample_dim: int,
                 n_cmps: int,
                 hidden_sizes_expert: Sequence[int],
                 hidden_sizes_ctxt_distr: Sequence[int],
                 init_exp: str = 'normal',
                 init_ctxt_distr: str = 'normal',
                 activation_exp: str = 'tanh',
                 activation_ctxt_distr: str = 'relu',
                 contextual_std: bool = False,
                 init_std: float = 1,
                 init_mean: float = 0,
                 policy_type: str = 'diag',
                 device: ch.device = "cpu",
                 dtype=ch.float32,
                 proj_type='kl'):
        self._ctxt_dim = ctxt_dim
        self._sample_dim = sample_dim
        self._cmps = []
        self.device = device
        self.cpu = True if device.type == 'cpu' else False
        self.dtype = dtype

        # expert information
        self.exp_policy_type = policy_type
        self.exp_proj_type = proj_type
        self.exp_init = init_exp
        self.init_ctxt_distr = init_ctxt_distr
        self.exp_hidden_sizes = hidden_sizes_expert
        self.hidden_sizes_ctxt_distr = hidden_sizes_ctxt_distr
        self.exp_activation = activation_exp
        self.activation_ctxt_distr = activation_ctxt_distr
        self.exp_contextual_std = contextual_std
        self.exp_init_std = init_std
        self.exp_init_mean = init_mean
        self._ctxt_distribution = []

        for i in range(n_cmps):
            self._cmps.append(
                get_policy_network(policy_type=policy_type, proj_type=proj_type, scaled_std=False, device=device,
                                   dtype=dtype, obs_dim=ctxt_dim, action_dim=sample_dim, init=init_exp,
                                   hidden_sizes=hidden_sizes_expert, activation=activation_exp,
                                   contextual_std=contextual_std, init_std=init_std, init_mean=init_mean,
                                   minimal_std=1e-5))
            self._ctxt_distribution.append(get_policy_network(policy_type='ctxt_softmax', obs_dim=ctxt_dim,
                                                              init=init_ctxt_distr,
                                                              hidden_sizes=hidden_sizes_ctxt_distr,
                                                              activation=activation_ctxt_distr, proj_type=None,
                                                              dtype=dtype,
                                                              device=device))
        self._cmps = [expert.to(self.device, self.dtype) for expert in self._cmps]
        self._ctxt_distribution = [ctxt_distr.to(self.device, self.dtype) for ctxt_distr in self._ctxt_distribution]
        self._log_weights = -ch.log(ch.ones(n_cmps, dtype=self.dtype, device=self.device) * n_cmps)[None, :]

    def sample(self, ctxts, gating_probs=None):
        if gating_probs is None:
            gating_probs = self.gating_probs(ctxts).detach()
        thresh = ch.cumsum(gating_probs, dim=1, dtype=self.dtype)
        thresh[:, -1] = ch.ones(ctxts.shape[0], dtype=self.dtype, device=self.device)
        eps = ch.rand(size=[ctxts.shape[0], 1], dtype=self.dtype, device=self.device)
        comp_idx_samples = np.argmax(get_numpy(eps) < get_numpy(thresh), axis=-1)
        samples = ch.zeros(size=(ctxts.shape[0], self._sample_dim), dtype=self.dtype, device=self.device)
        for i in range(self.num_components):
            ctxt_samples_cmp_i_idx = np.where(comp_idx_samples == i)[0]
            ctxt_samples_cmp_i = ctxts[ctxt_samples_cmp_i_idx, :]
            if ctxt_samples_cmp_i.shape[0] != 0:
                samples[ctxt_samples_cmp_i_idx, :] = self._cmps[i].sample(ctxt_samples_cmp_i, n=1)
        return samples.detach(), comp_idx_samples

    # pi(s|o) and pi(s)
    def log_cmp_m_ctxt_densities(self, ctxts):
        log_cmp_ctxt_densities = self.log_cmp_ctxt_densities(ctxts)
        # log sum exp
        exp_arg = log_cmp_ctxt_densities + self._log_weights
        log_marg_ctxt_densities = ch.logsumexp(exp_arg, dim=1)
        return log_cmp_ctxt_densities, log_marg_ctxt_densities

    # pi(s|o)
    def log_cmp_ctxt_densities(self, ctxts):
        n_comps = self.num_components
        log_probs = ch.zeros((ctxts.shape[0], n_comps), dtype=self.dtype, device=self.device)
        for i in range(n_comps):
            log_probs[:, i] = tensorize(self._ctxt_distribution[i](ctxts)).squeeze()
        return log_probs

    def log_gating_probs(self, ctxts):
        log_cmp_ctxt_densities, log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(ctxts)
        log_gating_probs = log_cmp_ctxt_densities + self._log_weights - log_marg_ctxt_densities[:, None]
        return log_gating_probs

    # pi(o|s)
    def gating_probs(self, ctxts):
        return ch.exp(self.log_gating_probs(ctxts))

    # pi(a|s) = log(sum_o pi(a|s,o) pi(o|s))
    def log_density(self, ctxts, samples):
        log_cmp_densities = self.log_cmp_densities(ctxts, samples)
        log_gating_probs = self.log_gating_probs(ctxts)
        exp_arg = log_cmp_densities + log_gating_probs
        log_density = log_sum_exp(exp_arg, axis=1)
        return log_density

    # sum_o pi(a|s,o) pi(o|s)
    def density(self, ctxts, samples):
        return ch.exp(self.log_density(ctxts, samples))

    def log_cmp_densities(self, ctxts, samples):
        n_comps = self.num_components
        log_probs = ch.zeros((ctxts.shape[0], n_comps), dtype=self.dtype, device=self.device)
        for i in range(n_comps):
            log_probs[:, i] = self._cmps[i].log_density(ctxts, samples)
        return log_probs

    def cmp_densities(self, ctxts, samples):
        return ch.exp(self.log_cmp_densities(ctxts, samples))

    # pi(o|a,s), pi(o|s)
    # return both because pi(o|s) is automatically calculated
    def log_responsibilities(self, ctxts, samples):
        log_cmp_ctxt_densities, log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(ctxts)
        log_gating_probs = log_cmp_ctxt_densities + self._log_weights - log_marg_ctxt_densities[:, None]
        log_cmp_densities = self.log_cmp_densities(ctxts, samples)
        log_model_density = ch.logsumexp(log_cmp_densities + log_gating_probs, dim=1)
        log_resps = log_cmp_densities + log_gating_probs - log_model_density[:, None]
        return log_resps, log_gating_probs

    def get_means(self, ctxts, cmp_idx):
        means = ch.zeros((cmp_idx.shape[0], self._sample_dim), dtype=self.dtype, device=self.device)
        for i in range(self.num_components):
            indices = np.where(i == cmp_idx)[0]
            if indices.shape[0] != 0:
                means[indices] = self.components[i](ch.atleast_2d(ctxts[indices]))[0]
        return means

    @property
    def components(self):
        return self._cmps

    @property
    def ctxt_distribution(self):
        return self._ctxt_distribution

    @property
    def num_components(self):
        return len(self._cmps)

    @property
    def ctxt_dim(self):
        return self._ctxt_dim

    @property
    def smpl_dim(self):
        return self._sample_dim

    @property
    def get_device(self):
        return self.device

    @property
    def get_dtype(self):
        return self.dtype

    def add_component(self):
        expert = get_policy_network(policy_type=self.exp_policy_type, proj_type=self.exp_proj_type, scaled_std=False,
                                    device=self.device,
                                    dtype=self.dtype, obs_dim=self._ctxt_dim, action_dim=self._sample_dim,
                                    init=self.exp_init,
                                    hidden_sizes=self.exp_hidden_sizes, activation=self.exp_activation,
                                    contextual_std=self.exp_contextual_std, init_std=self.exp_init_std,
                                    init_mean=self.exp_init_mean,
                                    minimal_std=1e-5)
        ctxt_distr = get_policy_network(policy_type='ctxt_softmax', obs_dim=self._ctxt_dim, init=self.init_ctxt_distr,
                                        hidden_sizes=self.hidden_sizes_ctxt_distr,
                                        activation=self.activation_ctxt_distr,
                                        proj_type=None, dtype=self.dtype, device=self.device)
        expert = expert.to(self.device, self.dtype)
        ctxt_distr = ctxt_distr.to(self.device, self.dtype)
        self._cmps.append(expert)
        self._ctxt_distribution.append(ctxt_distr)
        n_cmps = self.num_components
        self._log_weights = -ch.log(ch.ones(n_cmps, dtype=self.dtype, device=self.device) * n_cmps)[None, :]
        return self.num_components

    def remove_component(self, idx):
        raise NotImplementedError

    @staticmethod
    def create_from_config(config):
        dtype = ch.float64 if config['general']['dtype'] == "float64" else ch.float32
        device = ch.device("cuda:0" if not config['general']['cpu'] else "cpu")
        model = CtxtDistrNonLinExpMOE(ctxt_dim=config['environment']['c_dim'],
                                      sample_dim=config['environment']['a_dim'],
                                      n_cmps=config['general']['n_init_cmps'],
                                      hidden_sizes_expert=config['experts']['hidden_sizes_policy'],
                                      hidden_sizes_ctxt_distr=config['ctxt_distr']['hidden_sizes_ctxt_distr'],
                                      init_exp=config['experts']['initialization'],
                                      init_ctxt_distr=config['ctxt_distr']['initialization'],
                                      activation_exp=config['experts']['activation'],
                                      activation_ctxt_distr=config['ctxt_distr']['activation_ctxt_distr'],
                                      contextual_std=config['experts']['contextual_std'],
                                      init_std=config['experts']['init_std'],
                                      init_mean=config['experts']['init_mean'],
                                      policy_type=config['experts']['policy_type'],
                                      device=device,
                                      dtype=dtype,
                                      proj_type=config['trl']['proj_type'])
        return model, dtype, device
