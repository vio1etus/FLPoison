from fl.models.model_utils import model2vec, vec2model
from .algorithmbase import AlgorithmBase
from fl.algorithms import algorithm_registry


@algorithm_registry
class FedOpt(AlgorithmBase):
    def __init__(self, args, model, optimizer=None):
        super().__init__(args, model, optimizer)

    # for client
    def init_local_epochs(self):
        return self.args.local_epochs

    def get_local_update(self, **kwargs):
        global_weights_vec = kwargs['global_weights_vec']
        update = model2vec(self.model) - global_weights_vec
        return update

    # for server
    def update(self, aggregated_update, **kwargs):
        global_weights_vec = kwargs['global_weights_vec']
        global_w = global_weights_vec + aggregated_update
        vec2model(global_w, self.model)
        return global_w
