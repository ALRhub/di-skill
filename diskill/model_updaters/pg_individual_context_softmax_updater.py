"""
Inspired by https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/algorithms/pg/pg.py.
But written in the episodic RL case, I.e. major changes done.
"""

from typing import Union

import torch as ch
import time

from model_updaters.base_updater import BaseUpdater
from utils.buffer_sub_sets import PGBufferSubset
from utils.network_utils import get_optimizer
from utils.torch_utils import generate_minibatches, select_batch, tensorize


def _plot_entity(entity, dotplot=False):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkagg')
    plt.figure()
    style = 'x' if dotplot else None
    plt.plot(entity, style)


class PGIndividualCtxtSoftmaxUpdater(BaseUpdater):
    def __init__(self, optimizers,
                 epochs: int,
                 beta: float,
                 norm_advantages: bool,
                 clip_advantages: float = -1,
                 device: ch.device = "cpu",
                 dtype=ch.float64,
                 importance_ratio_clip: float = -1,
                 config=None,
                 max_grad_norm: Union[float, None] = 0):
        super(PGIndividualCtxtSoftmaxUpdater, self).__init__()
        self.optimizers = optimizers
        self.epochs = epochs
        self.beta = beta
        self.norm_advantages = False
        self.clip_advantages = clip_advantages  # >0 -> no adv. clipping
        self.cpu = True if device.type == 'cpu' else False
        self.device = device
        self.dtype = dtype
        self.importance_ratio_clip = importance_ratio_clip  # >0 -> no importance ratio clipping. If <0 -> PPO
        self.config = config
        self.alpha_tilde = self.config['general'].get('alpha_tilde', 0)
        self.clip_grad_norm = max_grad_norm  # if <0 -> no gradient norm clipping

        self.model = None

    def update_beta(self, new_beta):
        self.beta = new_beta

    def update(self, r_b_subset: PGBufferSubset, rewards, iw, curr_distr, idx):
        with ch.no_grad():
            ctxts = r_b_subset.ctxts
            glob_ctxts = r_b_subset.glob_ctxts
            old_logpacs = r_b_subset.ctxt_log_probs
            sampled_indices = r_b_subset.sample_indices

        policy_log = self.policy_step(ctxts, glob_ctxts, old_logpacs, rewards, curr_distr, sampled_indices, idx)
        return policy_log

    def policy_step(self, ctxts, glob_ctxts, old_logpacs, advantages, c_policy, sampled_indices, cmp_idx):
        log = {}
        start_time = time.time()

        tot_loss = tensorize(0., self.cpu, self.dtype)
        tot_policy_loss = tensorize(0., self.cpu, self.dtype)
        tot_entropy_loss = tensorize(0., self.cpu, self.dtype)
        tot_log_resps_loss = tensorize(0., self.cpu, self.dtype)
        tot_ratio = tensorize(0., self.cpu, self.dtype)
        log_resps = self.model.log_gating_probs(glob_ctxts)[:, cmp_idx].detach()

        for _ in range(self.epochs):
            batch_indices = generate_minibatches(ctxts.shape[0], 1)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, ctxts, glob_ctxts, old_logpacs, advantages, sampled_indices)
                b_ctxt, _, b_old_logpacs, b_adv, b_sampled_indices = batch

                new_logpacs = c_policy(b_ctxt)
                new_ctxt_glob_ctxts = ch.exp(c_policy(glob_ctxts)).squeeze()
                loss, ratio, new_logpacs, _, adv_std = self.surrogate_loss(b_adv.squeeze(),
                                                                           new_logpacs.squeeze(),
                                                                           b_old_logpacs)

                # rescale entropy due to normalization of the other obj:
                entr_coeff = self.beta / adv_std if self.norm_advantages else self.beta
                entropy_bonus = entr_coeff * c_policy.entropy(glob_ctxts).squeeze()
                log_resp_scaling = (self.beta - self.config['general']['alpha'] - self.alpha_tilde)
                log_resp_scaling = log_resp_scaling / adv_std if self.norm_advantages else log_resp_scaling
                log_resp_bonus = log_resp_scaling * new_ctxt_glob_ctxts * log_resps

                sum_loss = loss - entropy_bonus.mean() - log_resp_bonus.sum()
                self.optimizers[cmp_idx].zero_grad()
                # loss.backward()
                sum_loss.backward()
                if self.clip_grad_norm > 0:
                    ch.nn.utils.clip_grad_norm_(c_policy.parameters(), self.clip_grad_norm)
                self.optimizers[cmp_idx].step()

                tot_loss += loss.cpu().detach()
                tot_policy_loss += loss.cpu().detach()
                tot_entropy_loss -= entropy_bonus.mean().cpu().detach()
                tot_log_resps_loss -= log_resp_bonus.sum().cpu().detach()
                tot_ratio += ratio.mean().cpu().detach()

        n_updates = self.epochs  # * self.num_minibatches
        log['pol_time'] = time.time() - start_time
        log['tot_pol_update_loss'] = tot_loss / n_updates
        log['policy_loss'] = tot_policy_loss / n_updates
        log['entropy_loss'] = tot_entropy_loss / n_updates
        log['log_resps_loss'] = tot_log_resps_loss / n_updates
        log['mean_tot_ratio'] = tot_ratio / n_updates
        log['log_gatings'] = log_resps.cpu().detach()
        return log

    def surrogate_loss(self, advantages: ch.Tensor, new_logpacs: ch.Tensor, old_logpacs: ch.Tensor):
        """
        Computes the surrogate reward for IS policy gradient R(\theta) = E[r_t * A_t]
        Optionally, clamping the ratio (for PPO) R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
        Args:
            advantages: unnormalized advantages
            new_logpacs: Log probabilities from current policy
            old_logpacs: Log probabilities from old policy
        Returns:
            The surrogate loss as described above
        """
        adv_mean = None
        adv_std = None
        # Normalized Advantages
        if self.norm_advantages:
            adv_mean = advantages.mean().detach()
            adv_std = advantages.std().detach() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        if self.clip_advantages > 0:
            advantages = ch.clamp(advantages, -self.clip_advantages, self.clip_advantages)

        # Ratio of new probabilities to old ones
        ratio = (new_logpacs - old_logpacs).exp()

        surrogate_loss = ratio * advantages

        # PPO clipped ratio
        if self.importance_ratio_clip > 0:
            ratio_clipped = ratio.clamp(1 - self.importance_ratio_clip, 1 + self.importance_ratio_clip)
            surrogate_loss2 = ratio_clipped * advantages
            surrogate_loss = ch.min(surrogate_loss, surrogate_loss2)
        return -surrogate_loss.mean(), ratio.detach(), new_logpacs.detach(), adv_mean, adv_std

    def add_cmp(self, model, n_cmps_add):
        for k in reversed(range(n_cmps_add)):
            self.optimizers.append(get_optimizer(self.config['ctxt_distr']['optimizer_ctxt_distr'],
                                                 model.ctxt_distribution[k - 1].parameters(),
                                                 self.config['ctxt_distr']['lr_ctxt_distr'], eps=1e-8))
