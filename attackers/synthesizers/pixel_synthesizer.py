
import torch
from torchvision import transforms

from .synthesizer import Synthesizer


class PixelSynthesizer(Synthesizer):
    def __init__(self, args, trigger, **kwargs) -> None:
        # trigger_position (x, y) coordinates of the trigger in the image, or % of the image size
        # backdoor_location default is the bottom-left corner following badnets paper
        # , attack_model, target_label, poisoning_ratio, source_label
        self.args = args
        self.trigger = trigger
        self.set_kwargs(kwargs)

        trigger_height, trigger_width = self.trigger.shape[-2], self.trigger.shape[-1]
        # default backdoor position is the bottom-left corner
        self.trigger_position = (-trigger_height, -trigger_width)
        self.setup_trigger()

    def setup_trigger(self, trigger=None):
        """setup pixel trigger
        """
        if self.trigger.dtype != torch.float32:
            self.trigger = self.trigger.to(dtype=torch.float32)
        # get the tranformed trigger value
        norm_transform = transforms.Normalize(self.args.mean, self.args.std)

        # ToTensor will convert np array ranging from 0,255 to tensor 0,1, which should not be used for trigger value, 1
        # Multi-triggers, select the first one
        if len(self.trigger.shape) == 4:
            selected_trigger = self.trigger.expand(
                -1, self.args.num_channels, *self.trigger.shape[-2:])
        else:  # single trigger
            selected_trigger = self.trigger.expand(
                self.args.num_channels, *self.trigger.shape[-2:])

        self.trigger = norm_transform(selected_trigger)
