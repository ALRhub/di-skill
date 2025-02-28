"""
Inspired by https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/algorithms/pg/pg.py.
But written in the episodic RL case, I.e. major changes done.
"""

import time
from typing import Union

from model_updaters.base_updater import BaseUpdater
import torch as ch
from utils.buffer_sub_sets import PGBufferSubset
from utils.network_utils import get_optimizer
from utils.torch_utils import tensorize, generate_minibatches, select_batch


class CriticUpdater(BaseUpdater):

    def __init__(self, optimizers_critic: list, val_epochs: int, num_minibatches: int, config=None,
                 clip_vf: Union[float, None] = None, device: ch.device = "cpu", dtype=ch.float32):
        super(CriticUpdater, self).__init__()
        self.optimizers_critic = optimizers_critic
        self.val_epochs = val_epochs
        self.num_minibatches = num_minibatches
        self.config = config
        self.alpha = config['general']['alpha']
        self.clip_vf = clip_vf  # clipping the value function around last val -> PPO
        self.cpu = True if device.type == 'cpu' else False
        self.dtype = dtype

    def update(self, r_b_subset: PGBufferSubset, iw, curr_critic, idx, **kwargs) -> dict:
        ctxts = r_b_subset.ctxts
        rewards = r_b_subset.rewards  # Task rewards
        old_values = r_b_subset.old_values
        critic_log = self.critic_step(curr_critic, ctxts, rewards, old_values, idx)
        return critic_log

    def critic_step(self, curr_critic, ctxts, rewards, old_values, idx):
        log = {}
        start_time = time.time()
        vf_losses = tensorize(0., self.cpu, self.dtype)
        current_v = tensorize(0., self.cpu, self.dtype)
        shared_critic = curr_critic.shared
        for _ in range(self.val_epochs):
            splits = generate_minibatches(ctxts.shape[0], self.num_minibatches)

            # Minibatch SGD
            for indices in splits:
                batch = select_batch(indices, rewards, old_values, ctxts)

                sel_returns, sel_old_values, sel_obs = batch
                vs = curr_critic(sel_obs)

                vf_loss = self.value_loss(vs, sel_returns.squeeze(), sel_old_values)

                self.optimizers_critic[idx].zero_grad()
                vf_loss.backward()
                self.optimizers_critic[idx].step()
                vf_losses += vf_loss.cpu().detach()
                current_v += vs.mean().cpu().detach()

        n_updates = self.val_epochs * self.num_minibatches
        log['vf_time'] = time.time() - start_time
        log['vf_losses'] = vf_losses / n_updates
        log['current_v'] = current_v / n_updates
        return log

    def value_loss(self, values: ch.Tensor, returns: ch.Tensor, old_vs: ch.Tensor):
        """
        Computes the value function loss.

        When using GAE we have L_t = ((v_t + A_t).detach() - v_{t})
        Without GAE we get L_t = (r(s,a) + y*V(s_t+1) - v_{t}) accordingly.

        Optionally, we clip the value function around the original value of v_t

        Returns:
        Args:
            values: Current value estimates
            returns: Computed returns with GAE or n-step
            old_vs: Old value function estimates from behavior policy

        Returns:
            Value function loss as described above.
        """

        vf_loss = (returns - values).pow(2)

        if self.clip_vf > 0:
            # In OpenAI's PPO implementation, we clip the value function around the previous value estimate
            # and use the worse of the clipped and unclipped versions to train the value function
            vs_clipped = old_vs + (values - old_vs).clamp(-self.clip_vf, self.clip_vf)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = ch.max(vf_loss, vf_loss_clipped)
        return vf_loss.mean()

    def add_cmp(self, model, n_cmps_add):
        for k in reversed(range(n_cmps_add)):
            self.optimizers_critic.append(get_optimizer(self.config['critic']['optimizer_critic'],
                                                        model.components[k - 1].critic.parameters(),
                                                        self.config['critic']['lr_critic'], eps=1e-8))
