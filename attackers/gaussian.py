import numpy as np

from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'model_poisoning', 'non_omniscient')
class Gaussian(MPBase, Client):
    """
    [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17
    submit Gaussian noise as the update
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'noise_mean': 0, 'noise_std': 1}
        self.update_and_set_attr()

    def non_omniscient(self):
        noise = np.random.normal(
            loc=self.noise_mean, scale=self.noise_std, size=self.update.shape).astype(np.float32)
        return noise
