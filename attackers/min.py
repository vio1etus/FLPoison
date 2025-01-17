import numpy as np
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


class MinBase(MPBase, Client):
    """
    [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/) - NDSS '21
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {
            'gamma_init': 10.0, 'stop_threshold': 1e-5}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def omniscient(self, clients):
        # self.__class__.__name__ is the string-type subclass name when inherited. here is MinSum or MinMax
        attack_vec = Min(clients, self.__class__.__name__,
                         'unit_vec', self.gamma_init, self.stop_threshold)
        # repeat attack vector for all attackers
        return np.tile(attack_vec, (self.args.num_adv, 1))


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinMax(MinBase):
    """
    MinMax attack aims to find a malicious gradient, whose maximum distance from other benign gradient updates is smaller than the maximum distance between any two benign gradient updates via finding a optimal gamma
    """
    pass


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class MinSum(MinBase):
    """
    MinSum seeks a malicious gradient whose sum of distances from other benign gradient updates is smaller than the sum of distances of any benign gradient updates from other benign updates via finding a optimal gamma
    """
    pass


def get_metrics(metric_type):
    if metric_type == 'MinMax':
        def metric(x): return np.linalg.norm(x, axis=1).max()
    elif metric_type == 'MinSum':
        def metric(x): return np.square(np.linalg.norm(x, axis=1)).sum()
    return metric


def Min(clients, type, dev_type, gamma_init, stop_threshold):
    metric = get_metrics(type)
    # get benign updates and the mean of it
    benign_update = np.array(
        [i.update for i in clients if i.category == "benign"])
    benign_mean = np.mean(benign_update, 0)

    # select the type of deviation unit for subsequent perturbation, unit_vec by default
    if dev_type == 'unit_vec':  # unit vector
        deviation = benign_mean / np.linalg.norm(benign_mean)
    elif dev_type == 'sign':
        deviation = np.sign(benign_mean)
    elif dev_type == 'std':
        deviation = np.std(benign_update, 0)

    lamda, step, lamda_succ = gamma_init, gamma_init / 2, 0

    # get the upper bound of the metric value between the benign updates, so that the malicious update hidden between them being stealthy
    upper_bound = np.max([metric(benign_update-benign_update[i])
                         for i in range(len(benign_update))])

    # binary search for the optimal lamda to maximize the malicious update within the maximum distance
    while np.abs(lamda_succ - lamda) > stop_threshold:
        # perturb the mean update with perturbation vector
        mal_update = benign_mean - lamda * deviation
        # check the metric value between the malicious update and the benign updates
        mal_metric_value = metric(benign_update - mal_update)

        if mal_metric_value <= upper_bound:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda += step
        else:
            lamda -= step
        step /= 2
    # get the optimal malicious update by adding the perturbation vector
    mal_update = benign_mean - lamda_succ * deviation
    return mal_update
