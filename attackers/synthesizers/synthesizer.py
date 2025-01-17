
import random

import torch


class Synthesizer:
    """Synthesizer class for backdoor attacks.
    """

    def __init__(self, args, trigger, **kwargs) -> None:
        self.args = args
        self.trigger = trigger
        self.set_kwargs(kwargs)

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def backdoor_batch(self, images, labels, train=False, **kwargs):
        worker_id = kwargs.get('worker_id', None)

        if len(images.shape) == 4:  # batch
            len_images = images.shape[0]
            poisoned_idx_per_batch = self.setup_poisoned_idx(
                len_images, train)
            # 2. implement the main backdoor logic when the condition is met.
            for idx in range(len_images):
                if idx in poisoned_idx_per_batch:
                    # backdoor implantation
                    images[idx], labels[idx] = self.implant_backdoor(
                        images[idx], labels[idx], train=train, worker_id=worker_id)
        else:
            len_images = 1
            images, labels = self.implant_backdoor(
                images, labels, train=train, worker_id=worker_id)
            labels = torch.tensor(labels)
        return images, labels

    def implant_backdoor(self, image, label, **kwargs):
        # use the default implant_trigger function if not provided. LabelFlipping will provide the implant_trigger
        implant_trigger = kwargs.get(
            'implant_trigger', self.implant_trigger)
        if self.attack_model == "all2one":
            label = self.target_label
            implant_trigger(image, kwargs)
        elif self.attack_model == "random":
            label = torch.tensor(random.choice(
                range(self.args.num_classes)), dtype=torch.int64)
            implant_trigger(image, kwargs)
        elif self.attack_model == "all2all":
            label = self.args.num_classes - 1 - label
            implant_trigger(image, kwargs)
        elif self.attack_model == "targeted":
            # check if the source label is different from the target label
            assert self.source_label != self.target_label, self.args.logger.info(
                f"! Source label: {self.source_label}, Target label: {self.target_label} should not be equal")
            if label == self.source_label:
                label = self.target_label
                implant_trigger(image, kwargs)
        return image, label

    def implant_trigger(self, image, kwargs):
        # use the default trigger if not provided. DBA will provide the trigger
        trigger = kwargs.get('trigger', self.trigger)
        trigger_height, trigger_width = trigger.shape[-2], trigger.shape[-1]
        row_starter, column_starter = self.trigger_position
        # [-3:None] is equivalent to [-3:], 0 is not right
        image[..., row_starter: self.zero2none(row_starter+trigger_height),
              column_starter: self.zero2none(column_starter+trigger_width)] = trigger

    def setup_poisoned_idx(self, len_images, train, **kwargs):
        """
        if train is True, poison the images by batch by leveraging the self.poisoning_ratio
        if train is False, poison all images for evaluation
        if poisoning_len are provided, poison `poisoning_len` images in each batch
        """
        poisoning_len = kwargs.get('poisoning_len', None)
        if poisoning_len is None:
            # Setup indices for images to be poisoned in a batch. poison the images by batch by leveraging the self.poisoning_ratio
            batch_poisoning_ratio = self.poisoning_ratio if train else 1
            batch_poisoning_len = int(len_images * batch_poisoning_ratio)
        else:
            batch_poisoning_len = poisoning_len

        poisoned_idx_per_batch = random.sample(
            range(len_images), batch_poisoning_len)

        return poisoned_idx_per_batch

    def zero2none(self, x):
        """
        wrap the x to None if x is 0 for slice indexing in implant_backdoor, [-3:None] is equivalent to [-3:], 0 is not right
        """
        return x if x != 0 else None