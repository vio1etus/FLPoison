import math
import random
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class Bucketing(AggregatorBase):
    """
    [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt) - ICLR '22
    Bucketing aggregates the updates by first shuffling the updates, then dividing the updates into buckets of size bucket_size, and finally aggregating the updates in each bucket using the given aggregator, e.g., Krum, Mean, etc.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """"
        bucket_size (int): the size of each bucket, normally can be 2,5,10
        selected_aggregator (str): the aggregator used to aggregate the updates in each bucket
        """
        self.default_defense_params = {
            "bucket_size": 2, "selected_aggregator": "Krum"}
        self.update_and_set_attr()
        self.a_aggregator = aggregator_registry[self.selected_aggregator](
            args)
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        random.shuffle(updates)
        num_buckets = math.ceil(
            len(updates) / self.bucket_size)
        buckets = [updates[i:i + self.bucket_size]
                   for i in range(0, len(updates), self.bucket_size)]
        bucket_avg_updates = np.array(
            [np.mean(buckets[bucket_id], axis=0) for bucket_id in range(num_buckets)])

        return self.a_aggregator.aggregate(bucket_avg_updates)
