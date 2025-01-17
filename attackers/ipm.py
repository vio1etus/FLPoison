from global_utils import actor
from attackers.pbases.mpbase import MPBase
import numpy as np
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class IPM(MPBase, Client):
    """
    [Fall of empires: Breaking Byzantine-tolerant SGD by inner product manipulation](https://proceedings.mlr.press/v115/xie20a.html) - UAI '20
    IPM attack try to negate the inner product between the true benign gradient and the aggregated vector by submitting the scaled negative of the benign gradient
    IPM attack is a aggregator-specific attack,
    For Krum, Kurm-based Bulyan, and , the attack parameter should be appropriately small. 1. too big will be detected by the aggregator, 2. too small will not affect the final result too much
    For coordinate-wise median-like aggregators, such as GeometricMedian, direclty based on robust statistics, the attack parameter shoule big to break the robustness of the aggregator
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        # scale=0.1, 0.5, 1, 2, 100 all break FedAvg and GeometricMedian, small scale, such a 0.1, breaks Krum
        self.default_attack_params = {
            'scaling_factor': 0.1, "attack_start_epoch": None}
        self.update_and_set_attr()

    def omniscient(self, clients):
        if self.attack_start_epoch is not None and self.global_epoch <= 2 + self.attack_start_epoch:  # 62 start attack
            return None
        mean = np.mean(
            [i.update for i in clients if i.category == "benign"], axis=0)
        attack_vec = - self.scaling_factor * mean
        # repeat attack vector for all attackers
        return np.tile(attack_vec, (self.args.num_adv, 1))
