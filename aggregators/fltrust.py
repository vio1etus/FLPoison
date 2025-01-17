from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from aggregators.aggregator_utils import prepare_grad_updates, wrapup_aggregated_grads
from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from datapreprocessor.data_utils import dataset_class_indices, subset_by_idx
from fl.client import Client
from aggregators import aggregator_registry


@aggregator_registry
class FLTrust(AggregatorBase):
    """
    [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/abs/2012.13995) - NDSS '21
    FLTrust assumes that the server has a small benign dataset and trains a server benign model as the trust anchor, and computes the trust score as the cosine similarity between the client updates and the server models' update. The client updates are normalized by the server models' update, and then weighted by the trust score to compute the final aggregated update.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        num_sample (int): the number of samples to be used for server model training
        """
        self.default_defense_params = {"num_sample": 100}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"
        # init a client at server to maintain the server/root model
        train_dataset = kwargs['train_dataset']
        self._init_server_client(self.args, train_dataset)

    def _init_server_client(self, args, train_dataset):
        # sample a small benign train dataset with same distribution for server model training
        train_class_indices = dataset_class_indices(train_dataset)
        class_counts = [len(class_indices)
                        for class_indices in train_class_indices]
        cls_sample_size = [int(i * self.num_sample / sum(class_counts))
                           for i in class_counts]
        indices = np.zeros(sum(cls_sample_size), dtype=np.int64)
        start_idx = 0
        for cls_id in range(self.args.num_classes):
            selected_indices = np.random.choice(
                train_class_indices[cls_id], size=cls_sample_size[cls_id], replace=False)
            indices[start_idx:start_idx +
                    cls_sample_size[cls_id]] = selected_indices
            start_idx += cls_sample_size[cls_id]
        sampled_set = subset_by_idx(self.args, train_dataset, indices)
        self.server_client = Client(args, -1, sampled_set)
        self.server_client.set_algorithm(self.algorithm)

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        # get model parameters updates and gradient updates
        gradient_updates = prepare_grad_updates(
            self.args.algorithm, updates, self.global_model)

        # 1. server model training
        global_weights_vec = kwargs["global_weights_vec"]
        self.server_client.load_global_model(global_weights_vec)
        self.server_client.local_training()
        self.server_client.fetch_updates(benign_flag=True)

        # 2. get gradient update of server model
        raw_shape = self.server_client.update.shape
        root_grad_update = prepare_grad_updates(self.args.algorithm,  self.server_client.update.reshape(
            1, -1), self.global_model)
        root_grad_update.reshape(raw_shape)

        # 3. get the weighted cosine similarity between the client updates and the server client update as trust score
        TS = cosine_similarity(
            gradient_updates, root_grad_update.reshape(1, -1))

        # 4. apply relu to the similarity
        TS = np.maximum(TS, 0)
        TS /= np.sum(TS) + 1e-9  # weighted trust score

        # if the trust score is all zeros, set it to uniform, in case of server update deviation in fedsgd
        if not np.any(TS):
            TS = np.ones_like(TS) / len(TS)

        # 5. normalize the magnitudes of the client updates by the last global model
        normed_updates = gradient_updates / (np.linalg.norm(gradient_updates, axis=1).reshape(-1, 1) +
                                             1e-9) * np.linalg.norm(root_grad_update)

        agg_grad_updates = np.average(
            normed_updates, axis=0, weights=np.squeeze(TS))

        return wrapup_aggregated_grads(agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)
