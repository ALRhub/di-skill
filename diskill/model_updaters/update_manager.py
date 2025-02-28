from typing import Union

import numpy as np

from utils.buffer_sub_sets import BaseBufferSubset
from utils.functions import concatenate


class BaseMoEUpdater:
    def __init__(self, model, critics, expert_updater, ctxt_updater, weight_updater, ctxt_distr_updater, critic_updater):
        self.model = model
        self.critics = critics
        self.expert_updater = expert_updater
        self.ctxt_updater = ctxt_updater
        self.weight_updater = weight_updater
        self.ctxt_distr_updater = ctxt_distr_updater
        self.critic_updater = critic_updater
        self.ctxt_distr_updater.model = self.model

    def set_expert_updater(self, updater):
        self.expert_updater = updater

    def set_ctxt_updater(self, updater):
        self.ctxt_updater = updater

    def set_weight_updater(self, updater):
        self.weight_updater = updater

    def update_cmps(self, r_b_subset_list: list, rewards: list, iws: list, idxs: list, **kwargs) -> dict:
        results_all = {}
        for idx in idxs:
            results_all[idx] = self._update_cmp(r_b_subset_list[idx], rewards[idx], iws[idx], idx, **kwargs)
        return results_all

    def _update_cmp(self, r_b_subset: BaseBufferSubset, rewards: np.ndarray, iw: np.ndarray,
                    idx: int, **kwargs) -> dict:
        curr_distr = self.model.components[idx]
        return self.expert_updater.update(r_b_subset, rewards, iw, curr_distr, idx=idx, **kwargs)

    def update_context_distribution(self, r_b_subset_list: list, rewards: list, iws: list, idxs: list, **kwargs) -> dict:
        results_all = {}
        for idx in idxs:
            results_all[idx] = self._update_ctxt_cmp(r_b_subset_list[idx], rewards[idx], iws[idx], idx, **kwargs)
        return results_all

    def _update_ctxt_cmp(self, r_b_subset: BaseBufferSubset, rewards: np.ndarray, iw: np.ndarray,
                    idx: int, **kwargs) -> dict:
        curr_distr = self.model.ctxt_distribution[idx]
        return self.ctxt_distr_updater.update(r_b_subset, rewards, iw, curr_distr, idx=idx, **kwargs)

    def update_weights(self, rewards: np.ndarray) -> dict:  # TODO: Extend to IW version
        curr_distr = self.model.weight_distribution
        return {None: self.weight_updater.update(rewards, curr_distr)}

    def update_critics(self, r_b_subset_list: list, iws: list, idxs: list, **kwargs):
        results_all = {}
        if self.critics[0].shared or self.critics[0].global_critic:
            if len(idxs) != 0:
                r_b_subset, _, all_iws = self.merge_dataset(r_b_subset_list, None, iws, idxs)
                result = self._update_critic(r_b_subset, all_iws, idx=0, **kwargs)
                return {None: result}
        else:
            for idx in idxs:
                # if np.where(r_b_subset_list[idx].valid_ctxt_indices == True)[0].shape != 0:
                results_all[idx] = self._update_critic(r_b_subset_list[idx], iws[idx], idx, **kwargs)
        return results_all

    def _update_critic(self, r_b_subset: BaseBufferSubset, iw: np.ndarray, idx: int, **kwargs):
        curr_critic = self.critics[idx]
        return self.critic_updater.update(r_b_subset, iw, curr_critic, idx=idx, **kwargs)

    def update_beta(self, new_beta):
        self.ctxt_distr_updater.update_beta(new_beta)

    def merge_dataset(self,  r_b_subset_list: list, rewards: Union[list, None], iws: list, idxs: list):
        r_b_subset = r_b_subset_list[idxs[0]]
        all_rewards = rewards[idxs[0]] if rewards is not None else None
        all_iws = iws[idxs[0]]
        for j in range(len(idxs) - 1):
            r_b_subset.merge_buffer_subsets(r_b_subset_list[idxs[j + 1]])
            all_rewards = concatenate(all_rewards, rewards[idxs[j + 1]], axis=0) if all_rewards is not None else None
            all_iws = concatenate(all_iws, iws[idxs[j + 1]], axis=0)
        return r_b_subset, all_rewards, all_iws
