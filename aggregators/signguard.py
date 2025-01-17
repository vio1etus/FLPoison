from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
import random


@aggregator_registry
class SignGuard(AggregatorBase):
    """
    [Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering](https://arxiv.org/abs/2109.05872) - ICDCS '22
    SignGuard filters benign clients using a median norm-based threshold and performs clustering-based filtering based on the signs of the client weights. Finally, it clips them according to the median of their norms.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        lower_bound (float): the lower bound for the norm of the updates
        upper_bound (float): the upper bound for the norm of the updates
        selection_fraction (float): the fraction of the coordinates to be selected for clustering
        clustering (str): the clustering algorithm to be used (default: "MeanShift")
        random_seed (int): the random seed for reproducibility
        """
        self.default_defense_params = {
            "lower_bound": 0.1, "upper_bound": 3.0, "selection_fraction": 0.1, "clustering": "DBSCAN", "random_seed": 0}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 1. filtering based on the norm of the client weights
        S1_benign_idx, median_norm, client_norms = self.norm_filtering(
            gradient_updates)
        # 2. clustering based on the sign of the client weights
        S2_benign_idx = self.sign_clustering(gradient_updates)
        benign_idx = list(
            set(S1_benign_idx).intersection(S2_benign_idx))

        # 3. clip the benign gradients by median of norms
        grads_clipped_norm = np.clip(
            client_norms[benign_idx], a_min=0, a_max=median_norm)
        benign_clipped = (
            gradient_updates[benign_idx] / client_norms[benign_idx].reshape(-1, 1)) * grads_clipped_norm.reshape(-1, 1)

        return wrapup_aggregated_grads(benign_clipped, self.args.algorithm, self.global_model)

    def norm_filtering(self, gradient_updates):
        client_norms = np.linalg.norm(gradient_updates, axis=1)
        median_norm = np.median(client_norms)
        benign_idx = np.argwhere((client_norms > self.lower_bound * median_norm) & (
            client_norms < self.upper_bound * median_norm))
        return benign_idx.reshape(-1).tolist(), median_norm, client_norms

    def sign_clustering(self, gradient_updates):
        # 1. randomized coordinate selection
        num_para = gradient_updates.shape[1]
        num_selected = int(self.selection_fraction*num_para)
        idx = random.randint(0, int((1-self.selection_fraction)*num_para))
        # 2. extract positive, negative, and zero sign statistics
        randomized_weights = gradient_updates[:, idx:(
            idx+num_selected)]
        sign_grads = np.sign(randomized_weights)
        sign_type = {"pos": 1, "zero": 0, "neg": -1}

        def sign_feat(sign_type):
            sign_f = (sign_grads == sign_type).sum(
                axis=1, dtype=np.float32) / num_selected
            return sign_f / (sign_f.max() + 1e-8)
        sign_features = np.empty((self.args.num_clients, 3), dtype=np.float32)
        sign_features[:, 0] = sign_feat(sign_type["pos"])
        sign_features[:, 1] = sign_feat(sign_type["zero"])
        sign_features[:, 2] = sign_feat(sign_type["neg"])

        # 3. clustering based on the sign statistics
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(
                sign_features, quantile=0.5, n_samples=50)
            sign_cluster = MeanShift(bandwidth=bandwidth,
                                     bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            sign_cluster = DBSCAN(eps=0.05, min_samples=3)
        elif self.clustering == "KMeans":
            sign_cluster = KMeans(n_clusters=2, random_state=self.random_seed)

        sign_cluster.fit(sign_features)
        labels = sign_cluster.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        # 4. select the cluster with the majority of benign clients
        benign_label = np.argmax([np.sum(labels == i)
                                 for i in range(n_cluster)])
        benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label)]
        return benign_idx
