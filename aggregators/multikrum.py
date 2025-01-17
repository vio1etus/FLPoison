from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import L2_distances, krum_compute_scores
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class MultiKrum(AggregatorBase):
    """
    [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17

    Multi-Krum is a variant of Krum that selects the m updates with the smallest scores, rather than just the single update chosen by Krum, where the score is the sum of the n-f-1 smallest Euclidean distances to the other updates. Then it verages these selected updates to produce the final aggregated update.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        avg_percentage (float): the percentage of clients to be selected for averaging
        """
        self.default_defense_params = {
            "avg_percentage": 0.2, "enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        return multi_krum(
            updates, self.args.num_adv, avg_percentage=self.avg_percentage, enable_check=self.enable_check)


def multi_krum(updates, num_byzantine, avg_percentage, enable_check=False):
    """
    m_avg: select smallest m scores for averaging
    """
    num_clients = len(updates)
    m_avg = int(avg_percentage * num_clients)
    if enable_check:
        if num_clients <= 2 * num_byzantine+2:
            raise ValueError(
                f"num_byzantine should be meet 2f+2 < n, got 2*{num_byzantine}+2 >= {num_clients}."
            )
    # calculate euclidean distance between clients
    distances = L2_distances(updates)
    # calculate client i's score
    scores = [(i, krum_compute_scores(distances, i, num_clients, num_byzantine))
              for i in range(num_clients)]
    # sort index of client according to score
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return np.mean(updates[[sorted_scores[idx][0] for idx in range(m_avg)]], axis=0)
