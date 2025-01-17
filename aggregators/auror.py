from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from aggregators.aggregator_utils import prepare_grad_updates, prepare_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
import warnings
from fl.models.model_utils import ol_from_vector
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


@aggregator_registry
class Auror(AggregatorBase):
    """
    [Auror: Defending against poisoning attacks in collaborative deep learning systems](https://dl.acm.org/doi/10.1145/2991079.2991125) - ACSAC '16
    Auror cluster the coordinate value of the feature vector into 2 cluster, and determine the indices of indicative features by checking the distance between the cluster centers. Then, Auror clusters the indicative features to get majority cluster as benign ones for aggregation.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        indicative_threshold (float): Threshold for selecting indicative features based on cluster distance. A smaller value selects more features, increasing false positives. Suggested thresholds: 
        MNIST LeNet5 FedSGD lr=0.01: 1e-4; CIFAR10 ResNet18 FedSGD lr=0.01: 7e-4

        indicative_find_epoch (int): The first n epoch to find and determinate the indicative features, after that, the indicative features will be fixed
        """
        self.default_defense_params = {
            "indicative_threshold": 0.002, "indicative_find_epoch": 10}
        self.update_and_set_attr()
        self.epoch_cnt = 0
        # store the indices of indicative features of self.indicative_find_epoch
        self.indicative_idx = []
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # 1. find the indicative features (indices in feature vector) via 2-clustering with center distance threshold
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates according to the algorithm type
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # for the first 10 epoch, initialize and find the indicative_idx
        if self.epoch_cnt < self.indicative_find_epoch:
            self.indicative_idx = []
            # prepare the the (gradient) updates of output layers' parameter for each client
            self.ol_updates = np.array([
                ol_from_vector(
                    gradient_updates[cid], self.global_model, flatten=True, return_type='vector')
                for cid in range(self.args.num_clients)
            ])
            feature_dim = len(self.ol_updates[0])
            for feature_idx in range(feature_dim):
                feature_arr = self.ol_updates[:, feature_idx]
                kmeans = KMeans(n_clusters=2, random_state=0).fit(
                    feature_arr.reshape(-1, 1))
                centers = kmeans.cluster_centers_
                # self.args.logger.info(
                #     f"Global epoch {kwargs['global_epoch']}, abs(centers[0] - centers[1]):{abs(centers[0] - centers[1])}")
                if abs(centers[0] - centers[1]) >= self.indicative_threshold:
                    self.indicative_idx.append(feature_idx)
            # convert the indicative_idx of output layer to the whole model's indices
            self.indicative_idx = np.array(
                self.indicative_idx, dtype=np.int64) + len(gradient_updates[0]) - len(self.ol_updates[0])

        # 2. cluster the indicative features for anomaly detection
        indicative_updates = gradient_updates[:, self.indicative_idx]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(indicative_updates)
        labels = kmeans.labels_
        labels = labels[labels != -1]
        benign_label = 1 if np.sum(labels) > len(labels) / 2 else 0
        self.epoch_cnt += 1

        benign_grad_updates = gradient_updates[np.where(
            labels == benign_label)]
        return wrapup_aggregated_grads(benign_grad_updates, self.args.algorithm, self.global_model)
