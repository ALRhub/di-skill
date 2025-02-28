from abc import ABC, abstractmethod


class BaseUpdater(ABC):

    def __init__(self):
        self._updater = None

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """
        This function should return a dictionary with string keys.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def add_cmp(self, model, n_cmps_add: int):
        pass
