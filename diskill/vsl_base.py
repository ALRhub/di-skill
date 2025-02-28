import time
from typing import Any

import numpy as np
import torch as ch

from model_updaters.update_manager import BaseMoEUpdater
from utils.disp_logging import BaseLogger
from utils.functions import maybe_np
from utils.replay_buffer import BaseReplayBuffer
from utils.sampler import SampleManager
from utils.save_and_load import save_model_non_lin_ctxt_distr_moe, save_data
from utils.schedulers import RandomCMPAdder, UpdateScheduler


class VSLBase:
    def __init__(self, seed: int,
                 save2path: str,
                 alpha: float,
                 beta: float,
                 max_iter: int,
                 model: Any,
                 sample_manager: SampleManager,
                 n_samples_p_cmp: int,
                 batch_size: int,
                 replay_buffer: BaseReplayBuffer,
                 update_manager: BaseMoEUpdater,
                 verbose: int,
                 cmp_add_scheduler: RandomCMPAdder,
                 update_scheduler: UpdateScheduler,
                 test_every_it: int,
                 n_test_samples: int,
                 save_model_every_it: int,
                 dtype,
                 use_ch: bool = False,
                 all_configs=None,
                 log_verbosity: int = 1,
                 eval_max=False,
                 beta_scheduler = None
                 ):
        self.seed = seed
        self.save2path = save2path
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.model = model
        self.sample_manager = sample_manager
        self.n_samples_p_cmp = n_samples_p_cmp
        self.batch_size = batch_size
        self.rb = replay_buffer
        self.update_manager = update_manager
        self.update_scheduler = update_scheduler
        self.verbose = verbose
        self.cmp_add_scheduler = cmp_add_scheduler
        self.test_every_it = test_every_it
        self.n_test_samples = n_test_samples
        self.save_model_every_it = save_model_every_it
        self.dtype = dtype
        self.use_ch = use_ch
        self.all_configs = all_configs
        self.log_verbosity = log_verbosity
        self.eval_max = eval_max
        self.beta_scheduler = beta_scheduler
        self.disp_logger = BaseLogger()
        self.expert_val_fct = all_configs['critic']['use_critic']
        self.data = {'n_ex_samples': 0, 'n_ex_time_steps': 0, 'sampling_time': 0.}
        self.save_data = {'n_ex_samples': [], 'n_ex_time_steps': [], 'mean_rewards': [], 'max_rewards': [],
                          'mean_success': [], 'max_success': [], 'max_height': [], 'goal_dist': []}

    def train_iter(self, i):
        start_time = time.time()
        if self.beta_scheduler:
            new_beta = self.beta_scheduler.update(i)
            self.beta = new_beta
            self.update_manager.update_beta(new_beta)
        glob_ctxts = self.sample_manager.get_global_ctxts()
        add_cmps = self.cmp_add_scheduler.ask_to_add_cmp(i)
        if add_cmps:
            self.cmp_add_scheduler.add_cmps(self.model, self.update_manager, self.rb)
        n_sim_cmp_updates = self.cmp_add_scheduler.n_cmp_adds if add_cmps else self.update_scheduler.n_sim_cmp_updates
        self.update_scheduler.update_info(self.model.num_components, i, n_sim_cmp_updates)
        idx_mask = self.update_scheduler.get_cmp_idx_update_mask()

        # sampling goes here
        sampling_time = time.time()
        ex_samples = self.sample_manager.sample_from_env(self.model, self.n_samples_p_cmp, self.rb,
                                                         glob_ctxts, idx=idx_mask)
        self.data['sampling_time'] = time.time() - sampling_time
        self.data['n_ex_samples'] += ex_samples[0]
        self.data['n_ex_time_steps'] += ex_samples[1]

        r_b_subset_list = self.update_cmps(idx_mask, glob_ctxts)
        self.update_ctxt_cmps(idx_mask)
        with ch.no_grad():
            eval_ctxts = self.sample_manager.sample_contexts_only(5000)
            if self.log_verbosity == 1:
                for j in range(self.model.num_components):
                    # cmp wise logging
                    if r_b_subset_list[j] is not None:
                        ctxts = r_b_subset_list[j].ctxts
                        exp_entr = self.model.components[j].expected_entropy(ctxts).mean()
                    else:
                        exp_entr = None
                    self.data['exp_entropies/exp_entr_' + str(j)] = exp_entr
                    self.data['ctxt_cmp_entropies/ctxt_cmp' + str(j)] = self.model.ctxt_distribution[j].entropy(
                        eval_ctxts).mean()
                self.data['beta/beta'] = self.beta
            if i % self.test_every_it == 0:
                self._test_current_model()
            else:
                self.data['eval/success_mean'] = None
                self.data['eval/success_max'] = None
                self.data['eval/test_rew_mean'] = None
                self.data['eval/test_rew_max'] = None
        if i % self.save_model_every_it == 0 or i == self.max_iter - 1:
            it = None if i == self.max_iter - 1 else i
            self.save_model(it)
        if self.verbose:
            self.disp_logger.log_base_info(i, time.time() - start_time, self.data['eval/test_rew_mean'])
        return self.data

    def _test_current_model(self):
        with ch.no_grad():
            if self.eval_max:
                rewards_mean_max, infos_mean_max = self.sample_manager.test_model_mean_and_max(self.model,
                                                                                               self.n_test_samples)
                self.data['eval/test_rew_mean'] = np.mean(rewards_mean_max[0])
                self.data['eval/test_rew_max'] = np.mean(rewards_mean_max[1])
                if 'success' in infos_mean_max[0][0]:
                    success_var_exists = True
                    success_var = 'success'
                elif 'is_success' in infos_mean_max[0][0]:
                    success_var_exists = True
                    success_var = 'is_success'
                else:
                    success_var_exists = False
                if success_var_exists:
                    self.data['eval/success_mean'] = np.mean(np.array([elem[success_var][-1]
                                                                       for elem in infos_mean_max[0]]))
                    self.data['eval/success_max'] = np.mean(np.array([elem[success_var][-1]
                                                                      for elem in infos_mean_max[1]]))
                    self.save_data['mean_success'].append(self.data['eval/success_mean'])
                    self.save_data['max_success'].append(self.data['eval/success_max'])
                self.save_data['max_rewards'].append(self.data['eval/test_rew_max'])
            else:
                rewards, infos = self.sample_manager.test_model(self.model, self.n_test_samples, deterministic=True,
                                                                max_gating=False)
                self.data['eval/test_rew_mean'] = np.mean(rewards)
                if 'success' in infos[0]:
                    success_var_exists = True
                    success_var = 'success'
                elif 'is_success' in infos[0]:
                    success_var_exists = True
                    success_var = 'is_success'
                else:
                    success_var_exists = False
                if 'max_height' in infos[0]:
                    self.data['eval/max_height'] = np.mean(np.array([elem['max_height'][-1] for elem in infos]))
                    self.save_data['max_height'].append(self.data['eval/max_height'])
                    self.data['eval/goal_dist'] = np.mean(np.array([elem['contact_dist'][-1] for elem in infos]))
                    self.save_data['goal_dist'].append(self.data['eval/goal_dist'])
                if success_var_exists:
                    self.data['eval/success_mean'] = np.mean(np.array([elem[success_var][-1] for elem in infos]))
                    self.save_data['mean_success'].append(self.data['eval/success_mean'])
            self.save_data['n_ex_samples'].append(self.data['n_ex_samples'])
            self.save_data['n_ex_time_steps'].append(self.data['n_ex_time_steps'])
            self.save_data['mean_rewards'].append(self.data['eval/test_rew_mean'])

    def save_model(self, it):
        save_model_non_lin_ctxt_distr_moe(self.model, self.save2path,
                                          self.update_manager.expert_updater.optimizers_policy,
                                          self.update_manager.ctxt_distr_updater.optimizers, it, self.all_configs)
        save_data(self.save_data, self.save2path)

    def train(self):
        for i in range(self.max_iter):
            self.train_iter(i)

    def update_cmps(self, idx_mask, glob_ctxts):
        exp_rewards_list = []
        indices = []
        exp_update_indices = []
        iws = []
        r_b_subset_list = []
        sample_log_resps_list = []
        with ch.no_grad():
            for i in range(idx_mask.shape[0]):
                if idx_mask[i]:
                    r_b_subset = self.sample_manager.get_samples(self.model, i, self.batch_size, self.rb)
                    sample_log_resps = self.model.log_responsibilities(r_b_subset.ctxts, r_b_subset.samples)[0]
                    try:
                        rew = r_b_subset.advantages
                    except Exception:
                        rew = r_b_subset.rewards
                    expert_rews = self.calc_exp_rewards(rew, sample_log_resps[:, i])
                    exp_update_indices.append(i)
                    exp_rewards_list.append(expert_rews)
                    indices.append(i)
                    iws.append(np.ones(r_b_subset.ctxts.shape[0]) / r_b_subset.ctxts.shape[0])
                    r_b_subset_list.append(r_b_subset)
                    sample_log_resps_list.append(sample_log_resps[:, i])
                    self.rb.update(i)
                    ##### Logging
                    if self.log_verbosity == 1:
                        self.data['train_full_rew_exps/reward_expcmp_' + str(i)] = maybe_np(self.use_ch,
                                                                                            exp_rewards_list[i]).mean()
                        self.data['train_resps_exps/log_resps_expcmp_' + str(i)] = maybe_np(self.use_ch,
                                                                                            sample_log_resps_list[
                                                                                                i]).mean()
                        self.data['train_exps/reward_cmp_' + str(i)] = maybe_np(self.use_ch, r_b_subset.rewards).mean()
                        self.data['train_exps/advantage_cmp_' + str(i)] = maybe_np(self.use_ch,
                                                                                   r_b_subset.advantages).mean()
                else:
                    exp_rewards_list.append(None)
                    iws.append(None)
                    r_b_subset_list.append(None)
                    sample_log_resps_list.append(None)
                    ##### Logging
                    if self.log_verbosity == 1:
                        self.data['train_full_rew_exps/reward_expcmp_' + str(i)] = None
                        self.data['train_resps_exps/log_resps_expcmp_' + str(i)] = None
                        self.data['train_exps/reward_cmp_' + str(i)] = None
                        self.data['train_exps/advantage_cmp_' + str(i)] = None

        self.update_expert(indices, iws, r_b_subset_list, exp_rewards_list)
        if self.expert_val_fct:
            self.update_critic(exp_update_indices, iws, r_b_subset_list)
        return r_b_subset_list

    def update_ctxt_cmps(self, idx_mask):
        ctxt_cmps_rewards_list = []
        indices = []
        iws = []
        r_b_subset_list = []
        with ch.no_grad():
            for i in range(idx_mask.shape[0]):
                if idx_mask[i]:
                    r_b_subset = self.sample_manager.get_samples(self.model, i, self.batch_size, self.rb)
                    log_resps, _ = self.model.log_responsibilities(r_b_subset.ctxts, r_b_subset.samples)
                    try:
                        rew = r_b_subset.advantages_mean
                    except Exception:
                        rew = r_b_subset.rewards
                    try:
                        exp_entropies = self.model.components[i].expected_entropy(r_b_subset.ctxts)
                    except Exception:
                        exp_entropies = self.model.components[i].expected_entropy()
                    ctxt_cmp_rewards = self.calc_ctxt_distr_rewards(rew, log_resps[:, i], exp_entropies, None)
                    ctxt_cmps_rewards_list.append(ctxt_cmp_rewards)
                    indices.append(i)
                    iws.append(np.ones(r_b_subset.ctxts.shape[0]) / r_b_subset.ctxts.shape[0])
                    r_b_subset_list.append(r_b_subset)
                    self.rb.update(i)

                    ##### Logging
                    if self.log_verbosity == 1:
                        self.data['train_ctxt_distr_full_rew/reward_ctxtcmp_' + str(i)] = \
                            maybe_np(self.use_ch, ctxt_cmps_rewards_list[i]).mean()
                        self.data[fr'train_ctxt_distr_entropies/entropy_{i}'] = \
                            maybe_np(self.use_ch, self.model.ctxt_distribution[i].entropy(r_b_subset.ctxts))
                else:
                    ctxt_cmps_rewards_list.append(None)
                    iws.append(None)
                    r_b_subset_list.append(None)
                    if self.log_verbosity == 1:
                        self.data['train_ctxt_distr_full_rew/reward_ctxtcmp_' + str(i)] = None
                        self.data[fr'train_ctxt_distr_entropies/entropy_{i}'] = None
        self.update_ctxt_distribution(indices, iws, r_b_subset_list, ctxt_cmps_rewards_list)
        return r_b_subset_list

    def update_expert(self, idxs: list, iw: list, r_b_subset_list: list, rewards: list):
        logs = self.update_manager.update_cmps(r_b_subset_list, rewards, iw, idxs)
        if self.log_verbosity == 1:
            try:
                for j, l in enumerate(idxs):
                    self.data['trpl/policy_loss_' + str(l)] = logs[l]['policy_loss']
                    self.data['trpl/entropy_loss_' + str(l)] = logs[l]['entropy_loss']
                    self.data['trpl/trust_region_loss' + str(l)] = logs[l]['trust_region_loss']
                    self.data['trpl/kl_' + str(l)] = logs[l]['kl']
                    self.data['trpl/mean_constraint_' + str(l)] = logs[l]['mean_constraint']
                    self.data['trpl/max_mean_constraint_' + str(l)] = logs[l]['mean_constraint_max']
                    self.data['trpl/cov_constraint_' + str(l)] = logs[l]['cov_constraint']
                    self.data['trpl/max_cov_constraint_' + str(l)] = logs[l]['cov_constraint_max']
                    self.data['exp_update_infos/update_rejected_' + str(l)] = logs[l]['update_rejected']
                    self.data['exp_update_infos/exp_lr_' + str(l)] = logs[l]['exp_lr']
            except Exception:
                pass
        if self.verbose:
            self.disp_logger.log_info(logs, 'Exp. ')

    def update_critic(self, idxs: list, iw: list, r_b_subset_list: list):
        logs = self.update_manager.update_critics(r_b_subset_list, iw, idxs)
        if self.log_verbosity == 1:
            if self.update_manager.critics[0].shared or self.update_manager.critics[0].global_critic:
                for k, _ in enumerate(logs.keys()):
                    self.data[fr'critic/vf_loss{k}'] = logs[None]['vf_losses']
                    self.data[fr'critic/current_v_{k}'] = logs[None]['current_v']
            else:
                for k in logs.keys():
                    self.data[fr'critic/vf_loss{k}'] = logs[k]['vf_losses']
                    self.data[fr'critic/current_v_{k}'] = logs[k]['current_v']
        if self.verbose:
            self.disp_logger.log_info(logs, 'Critic ')

    def update_ctxt_distribution(self, idxs: list, iw: list, r_b_subset_list: list, rewards: list):
        logs = self.update_manager.update_context_distribution(r_b_subset_list, rewards, iw, idxs)
        if self.log_verbosity == 1:
            try:
                for j, l in enumerate(idxs):
                    self.data['ctxt_distr_train/policy_loss_' + str(l)] = logs[l]['policy_loss']
                    self.data['ctxt_distr_train/entropy_loss_' + str(l)] = logs[l]['entropy_loss']
                    self.data['ctxt_distr_train/log_resps_loss' + str(l)] = logs[l]['log_resps_loss']
                    self.data['ctxt_distr_train/mean_tot_ratio_' + str(l)] = logs[l]['mean_tot_ratio']
            except Exception:
                pass
        if self.verbose:
            self.disp_logger.log_info(logs, 'Context')

    def calc_exp_rewards(self, env_rewards: np.ndarray, log_resps: np.ndarray):
        return env_rewards + self.alpha * log_resps[:, None]

    def calc_ctxt_distr_rewards(self, env_rewards: np.ndarray, log_resps: np.ndarray, exp_entr: np.ndarray,
                                log_gating: np.ndarray):
        ### the log gatings are calculated and added to the loss in the PPO update step
        return self.calc_exp_rewards(env_rewards, log_resps) + self.alpha * exp_entr[:, None]  # + \
        # (self.beta - self.alpha) * log_gating[:, None]
