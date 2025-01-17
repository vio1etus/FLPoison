from copy import deepcopy
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class CenteredClipping(AggregatorBase):
    """
    [Learning from History for Byzantine Robust Optimization](https://arxiv.org/abs/2012.10333) - ICML '21
    It assumes worker use momentum, and the server aggregates the momentum updates by clipping that to the last round one, and then clip the aggregated update to a threshold.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedSGD"
        """
        norm_threshold (float): the threshold for clipping the aggregated update
        num_iters (int): the number of iterations for clipping the aggregated update
        """
        self.default_defense_params = {
            "norm_threshold": 100, "num_iters": 1}
        self.update_and_set_attr()
        self.momentum = None

    def aggregate(self, updates, **kwargs):
        if self.momentum is None:
            self.momentum = np.zeros_like(updates[0], dtype=np.float32)

        for _ in range(self.num_iters):
            self.momentum = (
                sum(self.clip(v - self.momentum)
                    for v in updates) / len(updates)
                + self.momentum
            )

        return deepcopy(self.momentum)

    def clip(self, v):
        scale = min(1, self.norm_threshold / np.linalg.norm(v, ord=2))
        return v * scale
