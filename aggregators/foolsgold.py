from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
import copy
import sklearn.metrics.pairwise as smp
from aggregators import aggregator_registry
from fl.models.model_utils import ol_from_model


@aggregator_registry
class FoolsGold(AggregatorBase):
    """
    [The Limitations of Federated Learning in Sybil Settings](https://www.usenix.org/conference/raid2020/presentation/fung) - RAID '20
    FoolsGold 
    It calculates the cosine similarity between the clients' accumulated updates and gets the max cosine similarity value of each client. Then, it re-weights the clients' updates based on the cosine similarity value.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        epsilon (float): a small value to avoid division by zero, log of zero, etc.
        topk_ratio (float): the ratio of the top-k largest absolute value of the output layer parameters of last global model to identify the indicative features
        """
        self.default_defense_params = {
            "epsilon": 1.0e-6, "topk_ratio": 0.1}
        self.update_and_set_attr()

        self.algorithm = "FedSGD"
        self.checkpoints = []

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs["last_global_model"]
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        feature_dim = len(gradient_updates[0])
        # weights for updates to re-weight for clients' updates
        wv = np.zeros((self.args.num_clients, 1), dtype=np.float32)
        # 1. record and sum the historical gradients
        # normalize updates of each client
        for cid in range(self.args.num_clients):
            cid_norm = np.linalg.norm(gradient_updates[cid])
            if cid_norm > 1:
                gradient_updates[cid] /= cid_norm
        self.checkpoints.append(copy.deepcopy(gradient_updates))
        sumed_updates = np.sum(self.checkpoints, axis=0)
        # 2. get the indicative features mask via top-k largest absolute value of the last global model
        ol_last_global_model = ol_from_model(
            self.global_model, flatten=False, return_type='vector')
        indicative_mask = self.get_indicative_mask(
            ol_last_global_model, feature_dim)

        # 3. calculate the cosine similarity (cs) between the clients' sum value and get the max cs value of each client
        cos_dist = smp.cosine_similarity(
            sumed_updates[:, indicative_mask == 1]) - np.eye(self.args.num_clients, dtype=np.float32)

        wv = self.pardoning(cos_dist)  # weight of updates
        agg_grad_updates = np.dot(gradient_updates.T, wv)
        return wrapup_aggregated_grads(agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)

    def pardoning(self, cos_dist):
        max_cs = np.max(cos_dist, axis=1) + self.epsilon
        # 4. pardoning
        # iterate i,j over the clients
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                if i == j:
                    continue
                if max_cs[i] < max_cs[j]:
                    cos_dist[i][j] *= max_cs[i] / max_cs[j]

        # diverse benign weights has smaller cosine similarity score, and should be re-weighted to have bigger weights
        wv = 1 - np.max(cos_dist, axis=1)
        wv = np.clip(wv, 0, 1)
        wv /= np.max(wv)
        wv[wv == 1] = .99

        # Logit function
        wv = np.log(wv / (1 - wv) + self.epsilon) + 0.5
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[wv < 0] = 0
        return wv

    def get_indicative_mask(self, ol_vec, feature_dim):
        class_dim, ol_feature_dim = ol_vec.shape[0], ol_vec.shape[1]
        ol_indicative_idx = np.zeros(
            (class_dim, ol_feature_dim), dtype=np.int64)  # index must be int or bool
        topk = int(class_dim * self.topk_ratio)
        for i in range(class_dim):  # class-wise top-k largest
            sig_features_idx = np.argpartition(ol_vec[i], -topk)[-topk:]
            ol_indicative_idx[i][sig_features_idx] = 1

        ol_indicative_idx = ol_indicative_idx.flatten()
        # extend ol_indicative_idx to feature_dim by padding zero before len(ol_indicative_idx)
        indicative_mask = np.pad(ol_indicative_idx,
                                 (feature_dim - len(ol_indicative_idx), 0), 'constant')
        return indicative_mask
