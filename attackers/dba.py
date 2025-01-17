import torch

from attackers import attacker_registry
from attackers.pbases.dpbase import DPBase
from fl.client import Client
from global_utils import actor

from .synthesizers import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning')
class DBA(DPBase, Client):
    """
    DBA introduce distributed backdoor attack, where each worker implant a local trigger at training time, and the global trigger is composed of all local triggers, launching the backdoor attack at inference time.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {"attack_model": "all2one", "scaling_factor": 100, "trigger_factor": [
            8, 2, 0], "poisoning_ratio": 0.32, "source_label": 2, "target_label": 7, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5, "attack_start_epoch": None}
        self.update_and_set_attr()
        self.algorithm = "FedOpt"
        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency, self.attack_start_epoch)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

        # self.num_benign_epoch, self.num_poison_epoch = 1, 10
        # self.benign_lr, self.poison_lr = 0.1, 0.05
        # self.benign_optimizer, self.optimizer = self.get_optimizer_scheduler(
        #     self.benign_lr)[0], self.get_optimizer_scheduler(self.poison_lr)[0]

    def define_synthesizer(self):
        # define four local triggers and global trigger is composed of local triggers
        # trigger_factor = size 4, gap 2, shift 0
        """
        ****   ****


        ****   ****
        """
        self.trigger_nums = 4
        self.trigger_size, self.gap, self.shift = self.trigger_factor
        self.trigger = torch.ones(
            (self.trigger_nums, 1, 1, self.trigger_size))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)
        # get synthesizer-transformed trigger for subsequently overwriting implant_distributed_backdoor
        self.trigger = self.synthesizer.trigger
        # implant the distributed backdoor via overwriting the synthesizer's implant_backdoor method
        self.implant_single_backdoor = self.synthesizer.implant_backdoor
        self.synthesizer.implant_backdoor = self.implant_distributed_backdoor

    def implant_distributed_backdoor(self, image, label, **kwargs):
        train, worker_id = kwargs['train'], kwargs['worker_id']
        # 2. implement the backdoor logic
        if train:
            # implant one of the four local trigger at the time of local training
            trigger_idx = worker_id % self.trigger_nums
            self.setup_trigger_position(trigger_idx)
            image, label = self.implant_single_backdoor(
                image, label, trigger=self.trigger[trigger_idx])
        else:
            # for global backdoor in server, embed all four local triggers to the image
            for trigger_idx in range(self.trigger_nums):
                self.setup_trigger_position(trigger_idx)
                image, label = self.implant_single_backdoor(
                    image, label, trigger=self.trigger[trigger_idx])
            #     print(self.trigger_position)
            # print(image)

        return image, label

    def setup_trigger_position(self, trigger_idx):
        """
        Following the paper, 4,2,0 for MNIST, 6,3,0 for CIFAR10
        """
        self.gap = 2
        self.shift = 0
        width = self.trigger.shape[-1]
        row_starter = (trigger_idx // 2) * (1+self.gap) + self.shift
        column_starter = (trigger_idx % 2) * (width + self.gap) + self.shift
        self.trigger_position = (row_starter, column_starter)

    # def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
    #     if self.global_epoch in self.poison_epochs:
    #         # benign training
    #         self.train_loader = self.get_dataloader(
    #             self.train_dataset, train_flag=True, poison_epochs=False)
    #         super().local_training(model, train_loader,
    #                                 self.benign_optimizer, criterion_fn, self.num_benign_epoch)
    #         # poison training
    #         self.train_loader = self.get_dataloader(
    #             self.train_dataset, train_flag=True, poison_epochs=True)
    #         super().local_training(model, train_loader,
    #                                    self.optimizer, criterion_fn, self.num_poison_epoch)
    #     else:
    #         return super().local_training()

    def non_omniscient(self):
        if self.global_epoch in self.poison_epochs:
            # scale
            scaled_update = self.global_weights_vec + self.scaling_factor * \
                (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        else:
            scaled_update = self.update
        return scaled_update
