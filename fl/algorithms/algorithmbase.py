from abc import ABC, abstractmethod


class AlgorithmBase(ABC):
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model  # same address with the model in the client or server
        # self.optimizer = optimizer

    @abstractmethod
    def init_local_epochs(self):
        pass

    @abstractmethod
    def get_local_update(self, *nouse):
        pass

    @abstractmethod
    def update(self, aggregated_update, *nouse):
        pass
