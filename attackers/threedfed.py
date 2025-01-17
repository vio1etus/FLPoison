
# !! Ongoing work, not finished yet

import copy
import logging
import numpy as np
import torch
from .badnets import BadNets
from attackers.pbases.mpbase import MPBase
from global_utils import actor, setup_logger
from attackers import attacker_registry
from typing import List
from fl.client import Client

# !!!Implement Unfinished Yet


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class ThreeDFed(MPBase, Client):
    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.indicators = {}
        self.num_decoy: int = 0  # num of decoy models
        alpha: List[float] = []
        # whether the server applies weak DP, if True, the attacker will use the weakDP strategy in the following epochs
        self.weakDP: bool = False

        log_file = f'./logs/attack_logs/{args.dataset}_{args.model}_{args.distribution}/{args.defense}.log'
        self.logger = setup_logger(
            "3dfed", log_file, level=logging.INFO)
        # For hybrid poisoning attacker, the attacker needs to define the synthesizer
        self.synthesizer = BadNets(
            args).synthesizer
        self.train_loader = self.get_dataloader(
            self.train_dataset, train_flag=True, poison_flag=True)

        # TODO: pass parameters and change others?
        self.poison_start_epoch, self.poison_end_epoch = 0

    def omniscient(self, clients):
        """After the local training, the attacker can perform the attack.
        """
        epoch = self.global_epoch
        if epoch == self.poison_start_epoch:
            self.design_indicator()
        elif epoch in range([self.poison_start_epoch+1, self.poison_end_epoch]):
            # 1. Read indicator feedback, Algorithm 3 in paper
            indicator_indices = 0
            accept = self.read_indicator(
                self, clients, indicator_indices)
            # Adaptive tuning the number of decoy models and self.alpha for backdoor training
            self.adaptive_tuning(accept)

        self.backdoor_update = copy.deepcopy(self.update)
        # 2. norm cliping, clip the backdoor updates to the norm size or predefined threshold of the benign updates
        super().local_training()
        super().fetch_updates()
        self.update = self.norm_clip(
            self.backdoor_update, self.update)
        # Find indicators
        self.design_indicator()

        # 3. Optimize noise masks
        self.optimize_noise_masks()
        # 4. Decoy model design

        pass

    def norm_clip(self, backdoor_update, benign_update):
        # 1. Get the norm of the benign update and backdoor update
        benign_norm, backdoor_norm = torch.norm(
            benign_update), torch.norm(backdoor_update)
        # 2. Clip the backdoor update to the norm size or predefined threshold of the benign update
        if backdoor_norm > benign_norm:
            backdoor_update = backdoor_update * \
                (benign_norm / backdoor_norm)
        # If the norm is so small, scale the norm to the magnitude of benign reference update
        scale_factor = min((benign_norm / backdoor_norm),
                           self.args.scaling_factor)
        return max(scale_factor, 1)*backdoor_update

    def design_indicator(self):
        total_devices = self.args.num_adv + self.num_decoy
        no_layer = 0
        # 1. Find indicators
        for i, data in enumerate(self.train_loader):
            # Compute gradient for backdoor batch with cross entropy loss
            images, labels = data[0].to(
                self.args.device), data[1].to(self.args.device)
            outputs = self.model(images)
            loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            # torch.autograd.grad will not add the gradient to the graph like backward(), so the gradient will not be accumulated
            # check x.requires_grad so that BatchNorm and Dropout layers will not be included
            grad = torch.autograd.grad(loss.mean(),
                                       [x for x in self.model.parameters() if
                                           x.requires_grad],
                                       create_graph=True
                                       )[no_layer]
            # Compute curvature (sencond order derivative)
            grad_sum = torch.sum(grad)
            curv = torch.autograd.grad(grad_sum,
                                       [x for x in self.model.parameters() if
                                        x.requires_grad],
                                       retain_graph=True
                                       )[no_layer]
            gradient += grad.detach().cpu().numpy()
            curvature += curv.detach().cpu().numpy()

        curvature = np.abs(curvature.flatten())
        # choose near zero curvature as indicators
        indicator_indices = np.argpartition(curvature, total_devices)[
            :total_devices]
        # TODO:

    def read_indicator(self, clients, indicator_indices):
        """get feedback which is the quotient of last global updates and current local updates on the same indicator indices, and mark accept, clipped, rejected for each client
        """
        accept, feedbacks = [], []

        # if previous epoch have already detect that the server is applying weakDP, then the attacker will use the weakDP strategy directly by discarding the indicator feedback function
        if self.weakDP:
            return accept

        # get indicator feedback
        for cid in range(len(clients)):
            feedbacks.append(clients[cid].update[indicator_indices] /
                             clients[cid].global_weights_vec[indicator_indices])

        # mark accept, clipped, rejected for each client for subsequent adaptive tuning
        threshold = 1e-4 if "MNIST" in self.args.dataset else 1e-5
        for feedback in feedbacks:
            if feedback > 1 or feedback < - threshold:
                self.weakDP = True
                break
            if feedback <= threshold:
                accept.append('r')      # r = rejected
            else:
                if feedback <= max(feedbacks) * 0.8:  # 0.5
                    accept.append('c')  # c = clipped
                else:
                    accept.append('a')  # a = accepted
        return accept

    def adaptive_tuning(self, accept):
        # if the server has already been detected applying weakDP, the attacker will use the weakDP strategy in the following epochs
        if self.weakDP:
            self.logger.warning("3DFed: disable adaptive tuning")
            for i in range(len(self.alpha)):
                self.alpha[i] = 0.1
            return self.alpha, self.num_decoy

        # adapt the number of decoy models, self.num_decoy
        group_size = self.args.adv_group_size
        accept_byzantine, accept_benign = accept[:
                                                 self.args.num_adv], accept[self.args.num_adv:]
        self.logger.warning(f'3DFed: acceptance status {accept}')
        self.num_decoy -= accept[self.args.num_adv:].count('a')
        self.num_decoy = max(self.num_decoy, 0)

        if 'a' not in accept_byzantine and 'c' not in accept_byzantine and accept_benign.count('a') <= 0:
            self.num_decoy += 1
        self.logger.info(f'3DFed: number of decoy models {self.num_decoy}')

        # Adaptively decide self.alpha, Algorithm 4 in paper
        # Divide malicious clients into serval groups, using different self.alpha fro backdoor training, so that the attacker can adaptively tune the attack-friendly self.alpha for each group according to indicator feedback
        alpha_candidate = []
        # TODO: if self.args.num_adv < group_size?
        group_num = int(self.args.num_adv / group_size)
        for i in range(group_num):
            count = accept[i*group_size:(i+1)*group_size].count('a')
            if count >= group_size * 0.8:
                alpha_candidate.append(self.alpha[i])
        alpha_candidate.sort()

        for i in range(group_num):
            # if the attacker has only one group
            if group_num <= 1:
                if len(alpha_candidate) == 0:
                    for j in range(len(self.alpha)):
                        self.alpha[j] = random.uniform(
                            self.args.noise_mask_alpha, 1.)
                break

            # if all the groups are accepted
            if len(alpha_candidate) == group_num:
                self.alpha[i] = random.uniform(
                    alpha_candidate[0], alpha_candidate[1])
            # if partial groups are accepted
            elif len(alpha_candidate) > 0:
                self.alpha[i] = random.uniform(
                    alpha_candidate[0], alpha_candidate[0]+0.1)
            # if no group is accepted
            else:
                self.alpha[i] = random.uniform(
                    self.args.noise_mask_alpha, 1.)  # += 0.1
        # revise the self.alpha range
        for i in range(len(self.alpha)):
            if self.alpha[i] >= 1:
                self.alpha[i] = 0.99
            elif self.alpha[i] <= 0:
                self.alpha[i] = 0.01

    def optimize_noise_masks(self):
        pass

    def decoy_model_design(self):
        pass
