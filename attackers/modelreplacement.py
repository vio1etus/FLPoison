import torch
from fl.client import Client
from attackers.pbases.mpbase import MPBase
from attackers.pbases.dpbase import DPBase
from global_utils import actor
from fl.models.model_utils import model2vec
from attackers import attacker_registry
from sklearn.metrics.pairwise import cosine_distances
from attackers.synthesizers.pixel_synthesizer import PixelSynthesizer


@attacker_registry
@actor('attacker', 'data_poisoning', 'model_poisoning', 'non_omniscient')
class ModelReplacement(MPBase, DPBase, Client):
    """
    [How to Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html) - AISTATS '20
    Model replacement attack, also known as constrain-and-scale attack and scaling attack, it first trains models with loss=normal_loss + anomaly_loss to avoid backdoor detection, then scales the update (X-G^t) by a factor gamma.
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        scaling_factor: estimated scaling factor, num_clients / global_lr, 50/1=50 in our setting
        alpha: the weight of the classification loss in the total loss
        """
        self.default_attack_params = {
            'scaling_factor': 50, "alpha": 0.5, "attack_model": "all2one",
            "poisoning_ratio": 0.32, "target_label": 6, "source_label": 3, "attack_strategy": "continuous", "single_epoch": 0, "poison_frequency": 5}
        self.update_and_set_attr()

        self.define_synthesizer()
        poison_epochs = self.generate_poison_epochs(
            self.attack_strategy, self.args.epochs, self.single_epoch, self.poison_frequency)
        self.train_loader = self.get_dataloader(
            train_dataset, train_flag=True, poison_epochs=poison_epochs)
        self.algorithm = "FedOpt"

    def define_synthesizer(self):
        # for pixel-type trigger, specify the trigger tensor
        self.trigger = torch.ones((1, 5, 5))
        self.synthesizer = PixelSynthesizer(
            self.args, self.trigger, attack_model=self.attack_model, target_label=self.target_label, poisoning_ratio=self.poisoning_ratio, source_label=self.source_label, single_epoch=self.single_epoch)

    def criterion_fn(self, y_pred, y_true, **kwargs):
        """rewrite the criterion function by adding an anomaly detection term, cosine distance between the local weights and the global weights
        # a L_class + (1-a) L_ano
        """
        # constrain: cosine distance between model2vec(self.model) and self.global_weights_vec
        cosine_dist = cosine_distances(model2vec(self.model).reshape(
            1, -1), self.global_weights_vec.reshape(1, -1))
        return self.alpha * torch.nn.CrossEntropyLoss()(y_pred, y_true) + (1-self.alpha) * torch.from_numpy(cosine_dist).to(self.args.device)

    def non_omniscient(self):
        # scale
        # gamma = self.args.num_clients/self.optimizer.param_groups[0]['lr'] # however, adversaries don't know num_clients
        # self.update = X - G^t
        scaled_update = self.global_weights_vec + self.scaling_factor * \
            (self.update - self.global_weights_vec) if self.args.algorithm == "FedAvg" else self.scaling_factor * self.update
        return scaled_update
