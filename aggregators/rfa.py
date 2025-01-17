import numpy as np
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry


@aggregator_registry
class RFA(AggregatorBase):
    """
    [Robust Aggregation for Federated Learning](https://ieeexplore.ieee.org/document/9721118) - ArXiv'19, TSP '22
    RFA (Geometric Median) replacing the weighted arithmetic mean aggregation with an approximate geometric median via the smoothed Weiszfeld algorithm.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        num_iters (int): the number of iterations to run the smoothed Weiszfeld algorithm
        epsilon (float): a small value to avoid division by zero
        """
        self.default_defense_params = {"num_iters": 3, "epsilon": 1.0e-6}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    def aggregate(self, updates, **kwargs):
        alphas = np.ones(len(updates), dtype=np.float32) / len(updates)
        # use the smoothed Weiszfeld algorithm to get the optimal geometric median vector of the updates
        return smoothed_weiszfeld(updates, alphas, self.epsilon, self.num_iters)


def smoothed_weiszfeld(updates, alphas, epsilon, num_iters):
    # v^0, the starting point of geometric median vector
    v = np.zeros_like(updates[0], dtype=np.float32)
    for _ in range(num_iters):
        denom = np.linalg.norm(updates - v, ord=2, axis=1)
        betas = alphas / np.maximum(denom, epsilon)
        v = np.dot(betas, updates) / betas.sum()
    return v
