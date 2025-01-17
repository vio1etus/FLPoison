import numpy as np
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

@aggregator_registry
class SimpleClustering(AggregatorBase):
    """
    Simple majority clustering based on gradient updates.
    """
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "clustering": "DBSCAN"}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(
                updates, quantile=0.5, n_samples=50)
            grad_cluster = MeanShift(bandwidth=bandwidth,
                                     bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            grad_cluster = DBSCAN(eps=0.05, min_samples=3)

        grad_cluster.fit(updates)
        labels = grad_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        # select the cluster with the majority of benign clients
        benign_label = np.argmax([np.sum(labels == i)
                                 for i in range(n_cluster)])
        benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label)]

        return wrapup_aggregated_grads(gradient_updates[benign_idx], self.args.algorithm, self.global_model)
