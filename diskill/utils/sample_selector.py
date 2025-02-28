import numpy as np

from utils.buffer_sub_sets import BaseBufferSubset, PGBufferSubset
from utils.replay_buffer import BaseReplayBuffer, PerCmpPGModReplayBuffer
from abc import ABC, abstractmethod
from collections import Counter


class BaseSampleSelector(ABC):

    @abstractmethod
    def select_samples(self, *args, **kwargs) -> BaseBufferSubset:
        raise NotImplementedError

    @staticmethod
    def select_sample_indices(replay_buffer: BaseReplayBuffer, n_last_samples: int) -> np.ndarray:
        # Take a continous batch from the buffer
        if replay_buffer.pos >= n_last_samples:
            return np.linspace(replay_buffer.pos - n_last_samples, replay_buffer.pos, endpoint=False, dtype=int,
                               num=n_last_samples)

        # Assemble batch from start and end of the buffer
        batch_rear_segment = np.linspace(0, replay_buffer.pos, endpoint=False, dtype=int, num=replay_buffer.pos)
        if replay_buffer.full:
            return np.concatenate((np.linspace(replay_buffer.buffer_size - n_last_samples + replay_buffer.pos,
                                               replay_buffer.buffer_size, endpoint=False, dtype=int,
                                               num=n_last_samples - replay_buffer.pos), batch_rear_segment))
        # Fill up batch with duplicates
        return np.concatenate(
            (batch_rear_segment, np.random.randint(0, replay_buffer.pos, n_last_samples - replay_buffer.pos)))


class NaiveSelector(BaseSampleSelector):

    def select_samples(self, replay_buffer: PerCmpPGModReplayBuffer, batch_size: int, **kwargs) -> BaseBufferSubset:
        idx = kwargs['idx']
        c_replay_buffer = replay_buffer.replay_buffer_list[idx]
        return c_replay_buffer.create_sample_set(self.select_sample_indices(c_replay_buffer, batch_size))
