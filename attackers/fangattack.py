from copy import deepcopy

from global_utils import actor
from attackers.pbases.mpbase import MPBase
import numpy as np
from aggregators.krum import krum
from attackers import attacker_registry
from fl.client import Client


@attacker_registry
@actor('attacker', 'omniscient')
class FangAttack(MPBase, Client):
    """
    [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/abs/1911.11815) - USENIX Security '20
    Fang's attack is a aggregator-specific attack with the knowledge of each adversaries (called partial knowledge in paper). If aggregator is unknown, assume one. here we assume krum.
    1. it treat the before-attack weights/gradients of attackers as benign updates, and get their mean as the update direction
    2. it crafts the attacker[0]'s weights/gradients to be selected by Krum via adding more malicious supporters around the crafted attacker's weights via binary search optimization
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'stop_threshold': 1.0e-5}
        self.update_and_set_attr()
        self.algorithm = "FedAvg"

    # after fetching updates from each client
    def omniscient(self, clients):
        # Here we use the fetched update for compatibility with various FL algorithms, instead of referring directly to weights or gradients
        # 1. get the mean of the f attackers' current before-attack weights/model, and estimate the update direction via sign(last_global_model - mean of current attacks' weights)
        before_attack_update = np.array(
            [c.update for c in clients if c.category == "attacker"])
        attacker_updates = np.zeros(
            (self.args.num_adv, len(self.update)), dtype=np.float32)
        est_direction = np.sign(np.mean(before_attack_update, axis=0))

        # global_weights_vec for fedavg comes from paper, 0 for fedsgd comes from fldetector code
        perturbation_base = self.global_weights_vec if self.args.algorithm == "FedAvg" else 0

        # 2. find the lambda value for the crafted attacker[0]'s weights to be selected by Krum via adding malicious supporters
        simulation_attack_number = 1
        assert self.args.num_adv > 1, "FangAttack requires more than 1 attacker"
        while (simulation_attack_number < self.args.num_adv):
            lambda_value = 1.0
            # formulate the before_attack weights (treated like benign updates while doing global aggregation) and the current crafted attacker's weights as updates for Krum selection, so that we can simulate some time to ensure the crafted attacker's weights are selected with only partial knowledge, controling the attackers' model updates.
            simulation_updates = np.empty(
                (self.args.num_adv+simulation_attack_number, len(self.update)), dtype=np.float32)
            simulation_updates[:self.args.num_adv] = before_attack_update

            # do-while loop
            while True:
                simulation_updates[self.args.num_adv: self.args.num_adv +
                                   simulation_attack_number] = perturbation_base - lambda_value * est_direction

                # send the before-attack models and crafted models to Krum selection from [0, self.args.num_adv + simulation_attack_number]
                min_idx = krum(simulation_updates,
                               simulation_attack_number, return_index=True)
                # if one of the crafted attacker (simulation_attack_number)'s weights are selected or the lambda value meets the stop threshold
                if min_idx >= self.args.num_adv or lambda_value <= self.stop_threshold:
                    break
                # reduce the lambda value by half and try again
                lambda_value *= 0.5
            simulation_attack_number += 1
            if min_idx >= self.args.num_adv:
                break

        attacker_updates[0] = perturbation_base - lambda_value * est_direction

        # 3. sample the other adversaries as supporter within the epsilon pertubation range
        for i in range(1, self.args.num_adv):
            # attacker_weights[i] = self.sample_vectors(epsilon = 0.01, attacker_weights[0], self.args.num_adv)[i-1]
            # or just let the crafted attacker's weights be the same as the first attacker's weights should be fine
            attacker_updates[i] = attacker_updates[0]
        return attacker_updates

    def sample_vectors(self, epsilon, w0_prime, num_byzantine):
        """Sphere generation method
        1. keep sampling random pertubation vectors and add it to the crafted attacker's weights,attacker_weights[0]
        2. if the pertubation vector is within the epsilon ball, then add attacker_weights[0] as the crafted attacker's weights for attacker_weights[i]
        return the other crafted attackers' updates around [attacker_updates[0]-epsilon, attacker_updates[0]+epsilon]
        """
        # store vectors that meet the conditions
        nearby_vectors = []
        while (len(nearby_vectors) < num_byzantine - 1):
            # generate random vector in the range of [w0_prime-epsilon, w0_prime+epsilon]
            random_vector = w0_prime + \
                np.random.uniform(-epsilon, epsilon, w0_prime.shape)
            if np.linalg.norm(random_vector - w0_prime) <= epsilon:
                nearby_vectors.append(random_vector)
        return np.stack(nearby_vectors, axis=0)
