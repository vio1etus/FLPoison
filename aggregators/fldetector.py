
from copy import deepcopy
from sklearn.cluster import KMeans
import numpy as np
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads


@aggregator_registry
class FLDetector(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {"window_size": 10, "start_epoch": 50}
        self.update_and_set_attr()
        self.algorithm = "FedOpt"
        # global weight W^t - W^(t-1), actually is g^t
        self.global_weight_diffs = []
        # global gradient g^t - g^(t-1), g can be seen as g(w), so that g(w)^t -g(w)^(t-1) = H (w^t - w^(t-1))
        self.global_grad_diffs = []
        self.last_global_grad = 0
        self.last_grad_updates = 0
        self.malicious_score = []
        self.init_model = None

    def aggregate(self, updates, **kwargs):
        # suppose the updates are gradients
        """
        weight_accumulator is pseduo-gradient updates
        global_model is the server global model
        """
        self.updates = updates
        self.global_model, self.current_epoch = deepcopy(
            kwargs['last_global_model']), kwargs['global_epoch']
        self.global_epoch = kwargs['global_epoch']

        if self.current_epoch <= self.start_epoch:
            # save the initial model for restart when outlier detected
            self.init_model = self.global_model

        _, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False)
        benign_idx = np.arange(len(gradient_updates))

        if self.current_epoch - self.start_epoch > self.window_size: # > 40 + 10
            hvp = self.LBFGS(self.global_weight_diffs, self.global_grad_diffs,
                             self.last_global_grad)
            # get the Euclidean distance of LBFGS-predicted gradient and gradient updates of each clients
            distance = self.get_pred_real_dists(
                self.last_grad_updates, gradient_updates, hvp)
            # self.args.logger.info(
            #     f"FLDetector: Global epoch {self.global_epoch}, distance: {distance}")
            self.malicious_score.append(distance)

        if len(self.malicious_score) > self.window_size:
            malicious_score = np.stack(
                self.malicious_score[-self.window_size:], axis=0)
            score = np.mean(malicious_score, axis=0)
            # Gap statistics
            if self.gap_statistics(score, num_sampling=20, K_max=10,
                                   n=self.args.num_clients)>= 2:
                # FLDetector's detection
                estimator = KMeans(n_clusters=2, n_init=10)
                estimator.fit(np.reshape(score, (score.shape[0], -1)))
                label_pred = estimator.labels_
                # cluster with larger average suspicious score as malicious
                benign_label = 1 if np.mean(score[label_pred == 0]) > np.mean(
                    score[label_pred == 1]) else 0
                benign_idx = np.argwhere(
                    label_pred == benign_label).squeeze()
                self.args.logger.info(
                        f"FLDetector Defense: Benign idx: {benign_idx}")
        agg_grad_update = np.mean(gradient_updates[benign_idx], axis=0)

        # save window-size record
        self.global_weight_diffs.append(agg_grad_update)
        self.global_grad_diffs.append(
            agg_grad_update - self.last_global_grad)
        if len(self.global_weight_diffs) > self.window_size:
            del self.global_weight_diffs[0]
            del self.global_grad_diffs[0]
        self.last_global_grad = agg_grad_update
        self.last_grad_updates = gradient_updates

        return wrapup_aggregated_grads(agg_grad_update, self.args.algorithm, self.global_model, aggregated=True)

    def get_pred_real_dists(self, last_grad_updates, gradient_updates, hvp):
        pred_grad = last_grad_updates + hvp
        distance = np.linalg.norm(pred_grad - gradient_updates, axis=1)
        distance = distance / np.sum(distance)
        return distance

    def LBFGS(self, S_k_list, Y_k_list, v):
        S_k_list = [i.reshape(-1, 1) for i in S_k_list]
        Y_k_list = [i.reshape(-1, 1) for i in Y_k_list]
        v = v.reshape(-1, 1)

        curr_S_k = np.concatenate(S_k_list, axis=1)
        curr_Y_k = np.concatenate(Y_k_list, axis=1)
        S_k_time_Y_k = np.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = np.matmul(curr_S_k.T, curr_S_k)

        R_k = np.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - np.array(R_k)
        sigma_k = np.matmul(Y_k_list[-1].T, S_k_list[-1]) / \
            (np.matmul(S_k_list[-1].T, S_k_list[-1]))
        D_k_diag = np.diag(S_k_time_Y_k)
        upper_mat = np.concatenate([sigma_k * S_k_time_S_k, L_k], axis=1)
        lower_mat = np.concatenate([L_k.T, -np.diag(D_k_diag)], axis=1)
        mat = np.concatenate([upper_mat, lower_mat], axis=0)
        mat_inv = np.linalg.inv(mat)

        approx_prod = sigma_k * v
        p_mat = np.concatenate([np.matmul(curr_S_k.T, sigma_k * v),
                                np.matmul(curr_Y_k.T, v)], axis=0)
        approx_prod -= np.matmul(np.matmul(np.concatenate([sigma_k *
                                                           curr_S_k, curr_Y_k], axis=1), mat_inv), p_mat)

        return approx_prod.squeeze()

    def gap_statistics(self, data, num_sampling, K_max, n):
        # normalize data
        data = normalize_data(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        gaps, s = [], []
        # adjust K_max，ensuring it won't exceed the number of data sample
        K_max = min(K_max, data.shape[0])

        for k in range(1, K_max + 1):
            # calculate the inertia of the real data
            kmeans = KMeans(n_clusters=k,n_init=10).fit(data)
            inertia = kmeans.inertia_

            # calculate the inertia of the fake data
            fake_inertia = []
            for _ in range(num_sampling):
                random_data = np.random.rand(n, data.shape[1])
                kmeans_fake = KMeans(n_clusters=k, n_init=10).fit(random_data)
                fake_inertia_i = kmeans_fake.inertia_
                fake_inertia.append(fake_inertia_i)

            # get Gap Statistic
            mean_fake_inertia = np.mean(fake_inertia)
            gap = np.log(mean_fake_inertia) - np.log(inertia)
            gaps.append(gap)

            # get standard deviation
            sd = np.std(np.log(fake_inertia))
            s.append(sd * np.sqrt((1 + num_sampling) / num_sampling))

        # choose the best num_cluster, k
        num_cluster = 0
        for k in range(1, K_max):
            if gaps[k - 1] - gaps[k] + s[k] >= 0:
                num_cluster = k+1
                break
        else:
            num_cluster = K_max
            print("FLDetector: No gap detected, No attack detected , return K_max")
        return num_cluster


def normalize_data(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
