from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators import aggregator_registry
from fl.models.model_utils import state2vec, vec2state


@aggregator_registry
class LASA(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {
            "norm_bound": 2, "sign_bound": 1, "sparsity": 0.3}  # CIRAR10/100 1,1, otherwise norm_bound=2
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def aggregate(self, updates, **kwargs):
        # load global model at last epoch
        num_clients = len(updates)
        self.global_model = kwargs['last_global_model']
        # state dict form of the updates with corresponding values not flattened
        dict_form_updates = []
        for i in range(num_clients):
            dict_form_updates.append(
                vec2state(updates[i], self.global_model, numpy=True))

        # 1. clip and scale based on median of norms of clients
        client_norms = np.linalg.norm(updates, axis=1)
        median_norm = np.median(client_norms)
        grads_clipped_norm = np.clip(client_norms, a_min=0, a_max=median_norm)
        grad_clipped = (updates / client_norms.reshape(-1, 1)
                        ) * grads_clipped_norm.reshape(-1, 1)

        dict_form_grad_clipped = [
            vec2state(grad_clipped[i], self.global_model, numpy=True) for i in range(num_clients)]

        # 1. Sparse each client's update with top-k largest strategy individually before aggregation
        for i in range(len(dict_form_updates)):
            dict_form_updates[i] = self.sparse_update(dict_form_updates[i])

        # for each layer
        key_mean_weight = {}
        for key in dict_form_updates[0].keys():
            if 'num_batches_tracked' in key:
                continue
            # 2. get the flattened gradient updates of the key
            key_flattened_updates = np.array([dict_form_updates[i][key].flatten()
                                              for i in range(num_clients)])

            # 3. magnitude filtering based on norm and MZ-score (Median Z-score)
            grad_l2norm = np.linalg.norm(key_flattened_updates, axis=1)
            S1_benign_idx = self.mz_score(grad_l2norm, self.norm_bound)

            # 4. direction filtering based on sign and  MZ-score (Median Z-score)
            layer_signs = np.empty(num_clients)
            for i in range(num_clients):
                sign_feat = np.sign(dict_form_updates[i][key])
                layer_signs[i] = 0.5 * np.sum(sign_feat) / \
                    np.sum(np.abs(sign_feat)) * (1 - self.sparsity)
            S2_benign_idx = self.mz_score(layer_signs, self.sign_bound)
            benign_idx = list(set(S1_benign_idx).intersection(S2_benign_idx))
            benign_idx = benign_idx if len(
                benign_idx) != 0 else list(range(num_clients))
            # layer-wise aggregation
            key_mean_weight[key] = np.mean(
                [dict_form_grad_clipped[i][key] for i in benign_idx], axis=0)

        return state2vec(key_mean_weight, numpy_flg=True)

    def sparse_update(self, update):
        """
        This function sparsifies the convlution and full-connection layer of updates of each client based on the top-k largest sparsification strategy
        """
        # 1. initialize the sparsity mask
        mask = {}
        for key in update.keys():
            if len(update[key].shape) == 4 or len(update[key].shape) == 2:
                # Need to change the dtype, but now only for testing
                mask[key] = np.ones_like(
                    update[key], dtype=np.float32)
        if self.sparsity == 0.0:
            return mask
        # 2. filter the top-k largest values for each key
        weight_abs = [np.abs(update[key])
                      for key in update.keys() if key in mask]
        # Gather all scores in a single vector and normalise
        all_scores = np.concatenate([value.flatten() for value in weight_abs])
        num_topk = int(len(all_scores) * (1 - self.sparsity))
        # top-k largest values
        kth_largest = np.partition(
            all_scores, -num_topk)[-num_topk]

        # 3. update the mask by setting the values smaller than the threshold to 0
        for key in mask.keys():
            # must be > to prevent acceptable_score is zero, leading to dense tensors
            mask[key] = np.where(
                np.abs(update[key]) <= kth_largest, 0, mask[key])

            # 4. apply the mask to the updates
            update[key].data *= mask[key]

        return update

    def mz_score(self, values, bound):
        med, std = np.median(values), np.std(values)
        for i in range(len(values)):
            values[i] = np.abs((values[i] - med) / std)
        return np.argwhere(values < bound).squeeze(-1)
