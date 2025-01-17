from attackers.pbases.dpbase import DPBase
from .synthesizers import Synthesizer
from global_utils import actor
from attackers import attacker_registry
from fl.client import Client
from functools import partial


@attacker_registry
@actor('attacker', 'data_poisoning')
class LabelFlipping(DPBase, Client):
    """
    LabelFlipping is label flipping attack with three substitution models for label flipping: random, target source label -> target label, inverse.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        attack_model: all2one, all2all, targeted, random
        source label: the label to be flipped
        target label: the label to be flipped to
        """
        self.default_attack_params = {
            'attack_model': 'targeted', 'source_label': 2, 'target_label': 7, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5, "poisoning_ratio": 0.32}
        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)

    def define_synthesizer(self):
        self.synthesizer = Synthesizer(
            self.args, None, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=1, source_label=self.source_label, single_epoch=self.single_epoch)

        # The label flipping attack only flips the source label, without following the poisoning rate.
        # if attack_model=targeted, all batch will be selected for poisoning
        train = False if self.attack_model == 'targeted' else True
        self.synthesizer.backdoor_batch = partial(
            self.synthesizer.backdoor_batch, train=train)
        # overwrite the implant_backdoor function
        self.synthesizer.implant_backdoor = partial(self.synthesizer.implant_backdoor,
                                                    implant_trigger=lambda image, kwargs: None)
