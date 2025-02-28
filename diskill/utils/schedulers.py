from typing import Any

import numpy as np
import torch as ch

from critic_models.vf_net import VFNet


class UpdateScheduler:
    def __init__(self, n_sim_cmp_updates: int, fine_tune_every_it: int, fine_tune_all_it: int):
        self.c_n_cmps = None
        self.cmp_idx_update_mask = None
        self.c_iter = 0
        self.n_sim_cmp_updates = n_sim_cmp_updates
        self.fine_tune_every_it = fine_tune_every_it
        self.fine_tune_all_it = fine_tune_all_it

    def update_info(self, n_cmps: int, c_it: int, n_sim_cmp_updates: int):
        self.c_n_cmps = n_cmps
        self.c_iter = c_it
        self.n_sim_cmp_updates = n_sim_cmp_updates

    def get_cmp_idx_update_mask(self):
        if self.c_iter >= self.fine_tune_all_it:
            return np.ones(self.c_n_cmps)
        elif self.c_iter % self.fine_tune_every_it == 0:
            return np.ones(self.c_n_cmps)
        else:
            idx_mask = np.zeros(self.c_n_cmps)
            idx_mask[-self.n_sim_cmp_updates:] = 1
            return idx_mask


class LinBetaScheduler:
    def __init__(self, init_beta, target_beta, max_it):
        self.init_beta = init_beta
        self.target_beta = target_beta
        self.max_it = max_it
        self._m = (self.target_beta - self.init_beta) / self.max_it
        self._t = init_beta

    def update(self, it):
        return self._m * it + self._t


class ExpBetaScheduler:
    def __init__(self, init_beta, target_beta, max_it):
        self.init_beta = init_beta
        self.target_beta = target_beta
        self.max_it = max_it
        self._m = (np.log(target_beta) - np.log(init_beta)) / (max_it - 0)
        self._t = (np.log(target_beta) - np.log(init_beta)) * 0 - np.log(init_beta)

    def update(self, it):
        return np.exp(self._m * it - self._t)


class RandomCMPAdder:

    def __init__(self, add_every_it: int, sample_dim: int, ctxt_dim: int, config, fine_tune_all_it: int,
                 n_cmp_adds: int = 1):
        self.add_every_it = add_every_it
        self.n_cmp_adds = n_cmp_adds
        self.fine_tune_all_it = fine_tune_all_it
        self.last_added_it = 0
        self.sample_dim = sample_dim
        self.ctxt_dim = ctxt_dim
        self.config = config
        self.add_after_it = self.config['general'].get('add_after_it', 0)

    def ask_to_add_cmp(self, it):
        if it >= self.fine_tune_all_it:
            return False
        if it != 0 and it >= self.add_after_it and it % self.add_every_it == 0:
            self.last_added_it = it
            return True
        else:
            return False

    def add_cmps(self, model: Any, update_manager, replay_buffer):
        critic_config = self.config['critic']
        for i in range(self.n_cmp_adds):
            model.add_component()
            update_manager.expert_updater.add_cmp(model, n_cmps_add=1)
            update_manager.ctxt_distr_updater.add_cmp(model, n_cmps_add=1)

            if critic_config['use_critic']:
                new_vf = VFNet(self.ctxt_dim, 1, init=critic_config['initialization'],
                               hidden_sizes=critic_config['hidden_sizes_critic'],
                               activation=critic_config['activation'],
                               shared=False, global_critic=False)
                device = ch.device("cuda:0" if not self.config['general']['cpu'] else "cpu")
                dtype = ch.float64 if self.config['general']['dtype'] == "float64" else ch.float32
                new_vf = new_vf.to(device, dtype)
                model.components[-1].set_critic(new_vf)
                update_manager.critic_updater.add_cmp(model, n_cmps_add=1)
                update_manager.critics.append(new_vf)
            replay_buffer.add_cmps(n_cmp_adds=1)

    @staticmethod
    def remove_component(model: Any, idx: int):
        raise NotImplementedError

    @staticmethod
    def remove_components(model: Any, idx_array: np.ndarray):
        # to remove all components at once
        # create another (mask) list with zeros (0=do not remove, 1=remove)
        remove_list = [0] * model.num_components
        for k in range(idx_array.shape[0]):
            remove_list[idx_array[k]] = 1

        completed_delete_list = False

        while not completed_delete_list:
            if len(remove_list) == 0:
                completed_delete_list = True
            else:
                completed_delete_list = True
                for k in range(len(remove_list)):
                    if remove_list[k] == 1:
                        completed_delete_list = False
            if not completed_delete_list:
                for k in range(len(remove_list)):
                    if remove_list[k] == 1:
                        del remove_list[k]
                        RandomCMPAdder.remove_component(model, k)
                        break
