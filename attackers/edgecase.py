import numpy as np
import torch
from attackers.pbases.dpbase import DPBase
from attackers.pbases.mpbase import MPBase
from datapreprocessor.edge_dataset import EdgeDataset
from global_utils import actor
from fl.models.model_utils import model2vec, vec2model
from attackers import attacker_registry
from .synthesizers import DatasetSynthesizer
from fl.client import Client

# TODO: test asr when pixel-based backdoor attack's bug is fixed


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class EdgeCase(MPBase, DPBase, Client):
    """
    [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) - NeurIPS '20
    Edge Case backdoor attack utilizes the edge-case samples viewed from the training dataset perspective (MNIST or CIFAR10) for edge-case backdoor embedding, and performs PGD and scaling attack to enhance the attack. Specifically, the attack steps include:
    1. edge-case data poisoning attack for MNIST and CIFAR10
        MNIST's edge dataset: label=7 images of ARDIS as label=7 of MNIST images; CIAFR10's edge dataset: southwest airline dataset as airplane of CIAFR10 Images
        self.args.target_label: the target label of the edge-case dataset
        for training: mix downsampled clean training dataset with edge-case dataset based on the poisoning_ratio
        for testing: edge-case dataset as full-poisoned test dataset and vanilla MNIST/CIFAR10 as vanilla test dataset
    2. L2 or L infinite norm-based PGD at the end of each local epoch
    2. model replacement attack, scaling attack
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        poisoning_ratio: ratio of edge data in the training dataset
        epsilon: Radius the l2 norm ball in PGD attack. For PGD with replacement, 0.25 for mnist, 0.083 for cifar10, coming from the paper
        projection_type: l_2 or l_inf
        l2_proj_frequency: projection frequency
        """
        self.default_attack_params = {
            "poisoning_ratio": 0.8, "epsilon": 0.25, "PGD_attack": True, "projection_type": "l_2", "l2_proj_frequency": 1, "scaling_attack": True, "scaling_factor": 50, "target_label": 1}
        self.update_and_set_attr()

        self.define_synthesizer()
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=True)
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        self.synthesizer = DatasetSynthesizer(
            self.args, self.train_dataset, EdgeDataset(self.args, self.target_label), self.poisoning_ratio)
        # initialize the poisoned train dataset and test dataset
        self.poisoned_set = self.synthesizer.get_poisoned_set(
            train=True), self.synthesizer.get_poisoned_set(train=False)

    def get_dataloader(self, dataset, train_flag, poison_epochs=None):
        # EdgeCase attack is this kind of attack using external prepared backdoor dataset
        poison_epochs = False if poison_epochs is None else poison_epochs
        data = self.poisoned_set[1 - train_flag] if poison_epochs else dataset
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=train_flag, num_workers=self.args.num_workers, pin_memory=True)
        while True:  # train mode for infinite loop with training epoch as the outer
            for images, targets in dataloader:
                yield images, targets
            if not train_flag:
                # test mode for test dataset
                break

    def step(self, optimizer, **kwargs):
        # PGD after step at each local epoch
        # normal step
        cur_local_epoch = kwargs["cur_local_epoch"]
        super().step(optimizer)

        # get the updated model
        model_update = model2vec(self.model)
        w_diff = model_update - self.global_weights_vec

        # PGD projection
        if self.projection_type == "l_inf":
            smaller_idx = np.less(w_diff, -self.epsilon)
            larger_idx = np.greater(w_diff, self.epsilon)
            model_update[smaller_idx] = self.global_weights_vec[smaller_idx] - self.epsilon
            model_update[larger_idx] = self.global_weights_vec[larger_idx] + self.epsilon
        elif self.projection_type == "l_2":
            w_diff_norm = np.linalg.norm(w_diff)
            if (cur_local_epoch % self.l2_proj_frequency == 0 or cur_local_epoch == self.local_epochs - 1) and w_diff_norm > self.epsilon:
                model_update = self.global_weights_vec + self.epsilon * w_diff / w_diff_norm

        # load the model_update to the model after PGD projection
        vec2model(model_update, self.model)

    def non_omniscient(self):
        # scaling attack (model replacement attacks)
        # non_omniscient function is after the get_local_update function
        if self.scaling_attack:
            scaled_update = self.global_weights_vec + self.scaling_factor * \
                (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        else:
            scaled_update = self.update
        return scaled_update
