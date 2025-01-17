import torch.nn.functional as F
import copy
from datapreprocessor.data_utils import Partition, dataset_class_indices, get_transform
import torch
from .labelflipping import LabelFlipping
from global_utils import actor
from fl.models.model_utils import model2vec, model2vec, vec2model
from attackers.pbases.mpbase import MPBase
from attackers.pbases.dpbase import DPBase
from attackers import attacker_registry
from fl.client import Client
import numpy as np


@attacker_registry
@actor('attacker', "data_poisoning", 'model_poisoning')
class AlterMin(MPBase, DPBase, Client):
    """
    [Analyzing Federated Learning Through an Adversarial Lens](https://arxiv.org/abs/1811.12470) - ICML '19

    two attack strategy: Alternating Minimization and Boosting 
    independent optimization for targeted objective and the stealth objectives
    1. stealth objectives: minimize the distance loss in code, normal CrossEntropy loss + distance loss for training on benign data
    2. if 100% attack success rate or step limit reached, otherwise do targeted objective: minimize the loss only on the malicious data
    Prodedures:
    1. targeted label flipping attack with only one malicious client
    2. training with modified loss function by adding distance loss between the malicious weights and the last global weights, which becomes stealth objectives: normal loss + distance loss
    3. update local model weights
    4. check target objective condition, if not, do targeted objective optimization by minimizing the loss only on the malicious data and do boosting for the malicious updates during this period
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        """
        attack_model: ["sample-targeted", "class-targeted"]
        poisoned_sample_cnt: number of poisoned samples,
        boosting_factor: boosting factor boosting_factor, variable `lambda` in paper
        rho: the weight of distance loss in the loss function
        benign_epochs: Benign training epochs for malicious agent
        malicous_epochs: malicious training epochs
        """
        self.default_attack_params = {"attack_model": "targeted", "source_label": 3, "target_label": 7, "poisoned_sample_cnt": 1,
                                      "boosting_factor": 10, "rho": 1e-4, "benign_epochs": 10, "malicous_epochs": 1}
        self.update_and_set_attr()
        self.algorithm = "FedOpt"
        # initialize the synthesizer for data loader
        self.define_synthesizer()
        self.init_poisoned_loader()
        self.train_loader = self.get_dataloader(
            self.train_dataset, train_flag=True, poison_epochs=False)

    def define_synthesizer(self,):
        # define a type of data poisoning attack
        dpa = LabelFlipping(
            self.args, self.worker_id, self.train_dataset, self.test_dataset)
        # note that other parameters are using the default values of LabelFlipping. If you want to change them, you can do it here
        dpa.source_label, dpa.target_label = self.source_label, self.target_label
        self.synthesizer = dpa.synthesizer

    def init_poisoned_loader(self):
        """Sample a subset of the train dataset for the poisoned data, and the rest for the benign data
        """
        # sample a small poisoned dataset from the test dataset
        posioned_indices = dataset_class_indices(
            self.test_dataset, class_label=self.source_label if self.attack_model == "targeted" else None)
        indices = np.random.choice(
            posioned_indices, self.poisoned_sample_cnt, replace=False)
        train_trans, _ = get_transform(self.args)
        poisoned_dataset = Partition(self.test_dataset, indices, train_trans)
        poisoned_dataset.poison_setup(self.synthesizer)
        self.poisoned_loader = torch.utils.data.DataLoader(
            poisoned_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

    def local_training(self):
        # 1. benign training with stealth objectives (normal+distance loss) on benign data
        train_acc, train_loss = super().local_training(
            criterion_fn=self.stealth_loss, local_epochs=self.benign_epochs)
        # 2. get asr (attack success rate) loss, if asr loss > 0, do targeted objective optimization
        asr, asr_loss = self.test(self.model, self.poisoned_loader)
        # optimize the target objective if malicious loss is not zero yet
        if asr_loss > 0.0:
            # 3. get weights before and after malicious training
            pre_train_weights = copy.deepcopy(model2vec(self.model))
            # malicious training with adversarial objectives on poisoned data
            super().local_training(train_loader=self.cycle(
                self.poisoned_loader), local_epochs=self.malicous_epochs)
            post_train_weights = copy.deepcopy(model2vec(self.model))

            # 5. check target objective condition. If not, do targeted objective optimization, explicit boosting
            boosted_weights = pre_train_weights + self.boosting_factor * \
                (post_train_weights - pre_train_weights)
            # 6. load the boosted weights to the model parameters
            vec2model(boosted_weights, self.model)
        return train_acc, train_loss

    def stealth_loss(self, y_pred, y_true):
        """rewrite the criterion function to include the distance loss
        """
        distance_loss = np.linalg.norm(
            model2vec(self.model) - self.global_weights_vec, 2)
        return torch.nn.CrossEntropyLoss()(y_pred, y_true) + self.rho * distance_loss

    def client_test(self, model=None, test_dataset=None, poison_epochs=False):
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)
        if poison_epochs:
            images, targets = map(lambda x: torch.cat(
                x, dim=0), zip(*self.poisoned_loader))
            # confidence of the target class
            test_acc, test_loss = self.malicious_samples_confidence(
                images, targets, model), 0
        else:
            test_loader = self.get_dataloader(
                test_dataset, train_flag=False, poison_epochs=False)
            test_acc, test_loss = self.test(model, test_loader)

        return test_acc, test_loss

    def malicious_samples_confidence(self, images, targets, model):
        model.eval()
        with torch.no_grad():
            # get the probabilities of each class by softmax
            probabilities = F.softmax(
                model(images.to(self.args.device)), dim=1)
            if len(images) == 1:  # confidence of single malicious sample
                target_confidences = probabilities[0, targets[0].item()].item()
            elif len(images) > 1:  # multiple malicious samples
                target_confidences = torch.sum(
                    targets == np.argmax(probabilities.cpu(), axis=1)) / len(images)

        return target_confidences
