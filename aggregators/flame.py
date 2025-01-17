from copy import deepcopy

import torch
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import hdbscan
from aggregators import aggregator_registry
from aggregators.aggregator_utils import normclipping, prepare_updates
from fl.models.model_utils import add_vec2model, model2vec


@aggregator_registry
class FLAME(AggregatorBase):
    """
    [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) - USENIX Security '22
    FLAME first clusters the cosine distance between client updates with hdbscan, then clips the benign gradients by the median of norms, and finally adds noise to meet the requirements of differential privacy.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.algorithm = "FedAvg"
        self.default_defense_params = {"gamma": 1.2e-5}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        model_updates, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model)
        benign_idx = self.cosine_clustering(model_updates)
        aggregated_model, median_norm = self.adpative_clipping(
            self.global_model, gradient_updates, benign_idx)
        self.add_noise2model(self.gamma * median_norm, aggregated_model)

        if self.args.algorithm == 'FedAvg':
            return model2vec(aggregated_model)
        else:
            return model2vec(aggregated_model) - model2vec(self.global_model)

    def cosine_clustering(self, model_updates):
        """
        clustering the cosine distance between client updates with hdbscan
        """
        cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                                  min_cluster_size=self.args.num_clients//2+1, min_samples=1, allow_single_cluster=True)
        cluster.fit(model_updates.astype(np.float64))
        # choose which cluster is benign
        return [idx for idx, label in enumerate(cluster.labels_) if label == 0]

    def adpative_clipping(self, last_global_model, gradient_updates, benign_idx):
        """
        clipping threshold is the median of l2 distance between last global model and current clients updates
        """
        # 1. get median of l2 norm
        median_norm = np.median(np.linalg.norm(gradient_updates, axis=1))
        # 2. clip the benign gradients by median of norms
        clipped_gradient_updates = normclipping(
            gradient_updates[benign_idx], median_norm)
        # 3. calculate the mean of clipped benign gradient updates and add them to the last global model for aggregation
        aggregated_gradient = np.mean(clipped_gradient_updates, axis=0)
        aggregated_model = add_vec2model(
            aggregated_gradient, last_global_model)
        return aggregated_model, median_norm

    def add_noise2model(self, noise_scale, model, only_weights=True):
        # add gaussian noise to the model ignoring bias and batch normalization layers
        model_state_dict = deepcopy(model.state_dict())
        for key, param in model_state_dict.items():
            if only_weights:
                if any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
                    continue
            std = noise_scale * param.data.std()
            noise = torch.normal(
                mean=0, std=std, size=param.size()).to(param.device)
            param.data += noise
        model.load_state_dict(model_state_dict)
