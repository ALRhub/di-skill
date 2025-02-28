from typing import Union, Tuple

import numpy as np
import torch as ch
import time

from model_updaters.base_updater import BaseUpdater
from projections.base_projection_layer import BaseProjectionLayer
from utils.buffer_sub_sets import PGBufferSubset
from utils.network_utils import get_optimizer
from utils.torch_utils import generate_minibatches, select_batch, tensorize
from copy import deepcopy, copy


# ch.autograd.set_detect_anomaly(True)


class PGUpdater(BaseUpdater):
    def __init__(self, optimizers_policy: list, projection: BaseProjectionLayer,
                 epochs: int, entropy_coeff: float, norm_advantages: bool, num_minibatches: int,
                 clip_advantages: float = -1, device: ch.device = "cpu", dtype=ch.float32,
                 importance_ratio_clip: float = -1, config=None, max_grad_norm: Union[float, None] = 0):
        super(PGUpdater, self).__init__()
        self.optimizers_policy = optimizers_policy
        self.base_lr = deepcopy(self.optimizers_policy[0].param_groups[0]['lr'])
        self.exp_update_step = [0] * len(optimizers_policy)
        self.projection = projection
        self.epochs = epochs
        self.entropy_coeff = entropy_coeff
        self.norm_advantages = norm_advantages
        self.num_minibatches = num_minibatches
        self.cpu = True if device.type == 'cpu' else False
        self.dtype = dtype
        self.config = config
        self.clip_advantages = clip_advantages  # >0 -> no adv. clipping
        self.importance_ratio_clip = importance_ratio_clip  # >0 -> no importance ratio clipping. If <0 -> PPO
        self.clip_grad_norm = max_grad_norm  # if <0 -> no gradient norm clipping
        self._reject_updates = config['trl']['rej_update_if_violated']

    def update(self, r_b_subset: PGBufferSubset, rewards, iw, curr_distr, idx):
        ctxts = r_b_subset.ctxts
        samples = r_b_subset.samples
        old_means = r_b_subset.old_means
        old_stds = r_b_subset.old_stds
        old_logpacs = r_b_subset.sample_log_probs

        policy_log = self.policy_step(ctxts, samples, old_logpacs, rewards, old_means, old_stds, curr_distr, idx)
        policy_log.update(self.log_metrics(curr_distr, ctxts, q=(old_means, old_stds)))
        self.exp_update_step[idx] += 1
        # TODO: include PAPI projection ?? -> see step function in TRL
        # TODO: add learning rate schedule -> see step funciton in TRL
        return policy_log

    def policy_step(self, ctxts, samples, old_logpacs, advantages, old_means, old_stds, c_policy, idx):
        log = {}
        start_time = time.time()
        q = (old_means, old_stds)

        # set initial entropy value in first step to calculate appropriate entropy decay
        # Note: we basically use the initial entropy of the first updated expert, i.e. the upcoming
        # experts probably start with another entropy which means that the schedule is not exactly correct
        if self.projection.initial_entropy is None:
            self.projection.initial_entropy = c_policy.entropy(q).mean()
        copy_of_c_policy = deepcopy(c_policy)

        tot_loss = tensorize(0., self.cpu, self.dtype)
        tot_policy_loss = tensorize(0., self.cpu, self.dtype)
        tot_entropy_loss = tensorize(0., self.cpu, self.dtype)
        tot_trust_region_loss = tensorize(0., self.cpu, self.dtype)
        for _ in range(self.epochs):
            batch_indices = generate_minibatches(ctxts.shape[0], self.num_minibatches)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, ctxts, samples, old_logpacs, advantages, q[0], q[1])
                b_ctxt, b_samples, b_old_logpacs, b_advantages, b_old_mean, b_old_std = batch
                b_q = (b_old_mean, b_old_std)

                p = c_policy(b_ctxt)
                proj_p = self.projection(c_policy, p, b_q, self.exp_update_step[idx])
                new_logpacs = c_policy.log_density_fixed(proj_p, b_samples)

                entropy_bonus = self.entropy_coeff * c_policy.entropy(proj_p)

                surrogate_loss, ratio, new_logpacs = self.surrogate_loss(b_advantages.squeeze(),
                                                                         new_logpacs, b_old_logpacs)

                # Trust region loss
                trust_region_loss = self.projection.get_trust_region_loss(c_policy, p, proj_p)

                # Total loss
                loss = surrogate_loss - entropy_bonus.mean() + trust_region_loss

                self.optimizers_policy[idx].zero_grad()
                loss.backward()
                if self.clip_grad_norm > 0:
                    ch.nn.utils.clip_grad_norm_(c_policy.parameters(), self.clip_grad_norm)
                self.optimizers_policy[idx].step()

                tot_loss += loss.cpu().detach()
                tot_policy_loss += surrogate_loss.cpu().detach()
                tot_entropy_loss -= entropy_bonus.mean().cpu().detach()
                tot_trust_region_loss += trust_region_loss.cpu().detach()

        # Sets the mean, if non-contextual, sets the cov if non-contextual
        if self._reject_updates:
            new_lr, update_rejected = self._update_policy(ctxts, q, c_policy, copy_of_c_policy, idx)
        else:
            update_rejected = False
            new_lr = self.optimizers_policy[idx].param_groups[0]['lr']
        if not update_rejected:
            self.update_old_policy(ctxts, q, c_policy, idx)
        n_updates = self.epochs * self.num_minibatches
        log['pol_time'] = time.time() - start_time
        log['tot_pol_update_loss'] = tot_loss / n_updates
        log['policy_loss'] = tot_policy_loss / n_updates
        log['entropy_loss'] = tot_entropy_loss / n_updates
        log['trust_region_loss'] = tot_trust_region_loss / n_updates
        log['update_rejected'] = update_rejected
        log['exp_lr'] = new_lr
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

        # Normalized Advantages
        if self.norm_advantages:
            advantages = (advantages - advantages.mean().detach()) / (advantages.std().detach() + 1e-8)

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

        return -surrogate_loss.mean(), ratio.detach(), new_logpacs.detach()

    def _update_policy(self, ctxts, q, c_policy, old_policy, idx):
        update_rejected = 0
        with ch.no_grad():
            p = c_policy(ctxts)
            constraint = ch.stack(self.projection.trust_region_value(c_policy, p, q), dim=-1).sum(-1).flatten()
            violation = constraint > self.projection.mean_bound + self.projection.cov_bound
            rel = violation.count_nonzero() / len(violation)
        if rel > 0.25:
            update_rejected = 1
            # Too many samples violate bounds
            # First schedule reduces lr by factor 0.8
            new_lr = np.clip(copy(self.optimizers_policy[idx].param_groups[0]['lr'] * 0.8), 1e-7, self.base_lr)
            # new_lr = np.clip(copy(self.optimizers_policy[idx].param_groups[0]['lr'] * 0.95), 1e-7, self.base_lr)
            self.optimizers_policy[idx].param_groups[0]['lr'] = new_lr
            c_policy.load_state_dict(old_policy.state_dict())
        else:
            # Second schedule increases lr by factor 1.01
            new_lr = np.clip(copy(self.optimizers_policy[idx].param_groups[0]['lr'] * 1.02), 1e-7, self.base_lr)
            self.optimizers_policy[idx].param_groups[0]['lr'] = new_lr
        return new_lr, update_rejected

    def update_old_policy(self, ctxts, q, c_policy, idx):
        """
        Set policy based on projection value without doing a regression.
        In non-contextual cases we have only one vector/matrix, so the projection is the same for all samples.
        Args:
            ctxts: batch observation to evaluate from
            q: old policy distributions
            c_policy: current policy
            idx: index of current expert
        Returns:
        """

        if self.projection.proj_type not in ["ppo", "papi"]:
            p = c_policy(ctxts)
            proj_p = self.projection(c_policy, p, q, self.exp_update_step[idx])

            if not c_policy.contextual:
                c_policy.set_mean(proj_p[0][0].detach())
            if not c_policy.contextual_std:
                c_policy.set_std(proj_p[1][0].detach())
            if c_policy.is_scaled:
                # TODO
                # self.policy.set_std_weight(w)
                raise NotImplementedError()

    def log_metrics(self, c_policy, ctxts, q):
        """
        Execute additional regression steps to match policy output and projection.
        The policy parameters are updated in-place.
        """
        metrics_dict = {}
        # get prediction before the regression to compare to regressed policy
        with ch.no_grad():
            p = c_policy(ctxts)
            constraints_initial_dict = self.projection.compute_metrics(c_policy, p, q)
        metrics_dict['kl'] = constraints_initial_dict['kl']
        metrics_dict['constraint'] = constraints_initial_dict['constraint']
        metrics_dict['mean_constraint'] = constraints_initial_dict['mean_constraint']
        metrics_dict['mean_constraint_max'] = constraints_initial_dict['mean_constraint_max']
        metrics_dict['cov_constraint'] = constraints_initial_dict['cov_constraint']
        metrics_dict['cov_constraint_max'] = constraints_initial_dict['cov_constraint_max']
        metrics_dict['entropy'] = constraints_initial_dict['entropy']
        metrics_dict['entropy_diff'] = constraints_initial_dict['entropy_diff']

        return metrics_dict

    def add_cmp(self, model, n_cmps_add):
        for k in reversed(range(n_cmps_add)):
            self.exp_update_step.append(0)
            self.optimizers_policy.append(get_optimizer(self.config['experts']['optimizer_policy'],
                                                        model.components[k - 1].parameters(),
                                                        self.config['experts']['lr_policy'], eps=1e-8))
