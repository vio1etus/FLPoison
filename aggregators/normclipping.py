from copy import deepcopy
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from aggregators.aggregator_utils import addnoise, normclipping, prepare_grad_updates, wrapup_aggregated_grads


@aggregator_registry
class NormClipping(AggregatorBase):
    """
    [Can You Really Backdoor Federated Learning](https://arxiv.org/abs/1911.07963) - NeurIPS '20
    It clips the norm of each client gradient updates by a threshold
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        norm_threshold (float): the threshold for clipping the norm of the updates
        """
        self.default_defense_params = {
            "weakDP": False, "norm_threshold": 3, "noise_mean": 0, "noise_std": 0.002}
        self.update_and_set_attr()

        self.algorithm = 'FedOpt'

    def aggregate(self, updates, **kwargs):
        # 1. prepare model updates, gradient updates, output layers of gradient updates
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        normed_updates = normclipping(gradient_updates, self.norm_threshold)
        # add noise to clients' updates
        if self.weakDP:
            normed_updates = addnoise(
                normed_updates,  self.noise_mean, self.noise_std)

        return wrapup_aggregated_grads(normed_updates, self.args.algorithm, self.global_model)
