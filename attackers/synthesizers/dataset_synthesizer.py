from .synthesizer import Synthesizer


class DatasetSynthesizer(Synthesizer):
    """
    For usage of Dataset Synthesizer, you need to override the get_dataloader method in dpbase.py to bypass the normal trigger embedding process
    """

    def __init__(self, args, train_dataset, poisoned_dataset, poisoned_ratio) -> None:
        self.args = args
        self.poisoned_dataset = poisoned_dataset
        self.poisoned_ratio = poisoned_ratio
        self.train_dataset = train_dataset

    def get_poisoned_set(self, train):
        """
        return poisoned dataset
        """
        if train:
            return self.poisoned_dataset.mix_trainset(self.train_dataset, self.poisoned_ratio)
        else:
            return self.poisoned_dataset.get_poisoned_testset()
