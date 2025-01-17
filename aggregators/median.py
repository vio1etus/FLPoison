from aggregators.aggregatorbase import AggregatorBase
import numpy as np

from aggregators import aggregator_registry


@aggregator_registry
class Median(AggregatorBase):
    """
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Coordinated Median computes the median of the updates coordinate-wisely.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def aggregate(self, updates, **kwargs):
        return np.median(updates, axis=0)
