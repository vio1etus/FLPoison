from copy import deepcopy
from aggregators.aggregator_utils import addnoise, prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class CRFL(AggregatorBase):
    """
    [CRFL: Certifiably Robust Federated Learning against Backdoor Attacks](http://proceedings.mlr.press/v139/xie21a/xie21a.pdf)
    CRFL apply parameters clipping and perturbing to mean aggregated update
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = 'FedOpt'
        self.default_defense_params = {
            "norm_threshold": 3, "noise_mean": 0, "noise_std": 0.001}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        # 1. prepare model updates, gradient updates, output layers of gradient updates
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 1. aggregate the gradient updates
        agg_update = np.mean(gradient_updates, axis=0)
        # 2. norm clip the updates
        normed_agg_update = agg_update * \
            min(1, self.norm_threshold / (np.linalg.norm(agg_update)+1e-10))

        # 3. add gaussian noise, note that the noise should be float32 to be consistent with the future torch dtype
        return wrapup_aggregated_grads(addnoise(normed_agg_update,  self.noise_mean, self.noise_std), self.args.algorithm, self.global_model, aggregated=True)
