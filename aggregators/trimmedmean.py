from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class TrimmedMean(AggregatorBase):
    """
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Trimmed Mean exludes the smallest and largest beta fraction coordiantes of the updates and averages the rest coordiantes.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        beta (float): fraction of updates to exclude, both from the top and the bottom
        """
        self.default_defense_params = {"beta": 0.1}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        return trimmed_mean(updates, self.beta)


def trimmed_mean(updates, filter_frac):
    num_excluded = int(filter_frac * len(updates))
    smallest_excluded = np.partition(
        updates, kth=num_excluded, axis=0)[:num_excluded]
    biggest_excluded = np.partition(
        updates, kth=-num_excluded, axis=0)[-num_excluded:]

    # fast way: add and substract. here directly add the negative values of smallest_excluded and biggest_excluded for counterbalance
    weights = np.concatenate(
        (updates, -smallest_excluded, -biggest_excluded)).sum(0)
    weights /= len(updates) - 2 * num_excluded
    return weights
