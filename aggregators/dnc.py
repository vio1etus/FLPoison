from copy import deepcopy
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry


@aggregator_registry
class DnC(AggregatorBase):
    """
    [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21
    DnC subsamples the parameters and projects the gradients to the top right singular eigenvector for outlier scores. It then selects the k-smallest scores as benign clients' indices.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        subsample_frac (float): the fraction of parameters to subsample for dimensionality reduction
        num_iters (int): the number of iterations to perform the outlier detection
        fliter_frac (float): the fraction of adversaries to filter out as attackers
        """
        self.default_defense_params = {
            "subsample_frac": 0.2, "num_iters": 5, "fliter_frac": 1.0}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        num_param = gradient_updates.shape[1]
        benign_idx = set(range(self.args.num_clients))
        for _ in range(self.num_iters):
            # 1. subsample a fraction of the parameters for dimensionality reduction
            param_idx = np.random.choice(
                np.arange(num_param), int(self.subsample_frac * num_param), replace=False)
            # set of gradients subsampled using indices in idx
            sampled_grads = gradient_updates[:, param_idx]
            # 2. center the sampled_grads to their mean
            mu = np.mean(sampled_grads, axis=0)
            # get centered input gradients
            centered_grads = sampled_grads - mu
            # 3. project the centered gradients to the top right singular eigenvector for outliner scores
            # get the top right singular eigenvector
            _, _, V = np.linalg.svd(centered_grads, full_matrices=False)
            v = V[0, :]
            # Compute outlier scores
            score = np.dot(centered_grads, v)**2
            # get k-smallest scores as begnin clients' indices
            k = int(self.args.num_clients-self.fliter_frac*self.args.num_adv)
            dnc_idx = np.argpartition(score, k, axis=0)[:k] if k != len(
                score) else np.arange(len(score))
            benign_idx = benign_idx.intersection(set(dnc_idx))

        return wrapup_aggregated_grads(gradient_updates[list(benign_idx)], self.args.algorithm, self.global_model)
