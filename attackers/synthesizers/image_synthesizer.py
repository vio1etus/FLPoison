from torchvision import transforms
from PIL import Image
from .synthesizer import Synthesizer


class ImageSynthesizer(Synthesizer):
    def __init__(self, args, trigger_path, trigger_size, attack_model, target_label, poisoning_ratio, source_label) -> None:
        # trigger_position (x, y) coordinates of the trigger in the image, or % of the image size
        # backdoor_location default is the bottom-left corner following badnets paper
        self.args = args
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.poisoning_ratio = poisoning_ratio
        self.source_label = source_label
        self.attack_model = attack_model
        

        self.setup_trigger()

    def setup_trigger(self, trigger=None):
        """accept a real-world image as the trigger
        """
        image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)])

        # check if grey or colorful images by dimensions, (Batch, Channel, H, W) or (Batch, 1, H, W)
        # mode = 'L' if self.images.shape[1] == 1 else 'RGB'
        mode = 'RGB' if self.args.dataset == "CIFAR10" else 'L'
        # RGBA is default; MNIST is grey, mode='L'; CIFAR10 is color, mode='RGB'
        trigger_img = Image.open(self.trigger_path).convert(mode)
        trigger_size = (self.trigger_size, self.trigger_size)
        trigger_img = trigger_img.resize(
            (trigger_size[0], trigger_size[1]))
        self.trigger = image_transform(trigger_img)
        # default backdoor position is the bottom-left corner
        self.trigger_position = (-trigger_size[0], -trigger_size[1])
