from fl.models.model_utils import model2vec, vec2model
from .algorithmbase import AlgorithmBase
from fl.algorithms import algorithm_registry


@algorithm_registry
class FedAvg(AlgorithmBase):
    def __init__(self, args, model, optimizer=None):
        super().__init__(args, model, optimizer)

    # for client
    def init_local_epochs(self):
        return self.args.local_epochs

    def get_local_update(self, **kwargs):
        update = model2vec(self.model)
        return update

    # for server
    def update(self, aggregated_update, **kwargs):
        # load parameters to model
        vec2model(aggregated_update, self.model)
        return aggregated_update
