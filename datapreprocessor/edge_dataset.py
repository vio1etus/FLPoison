import copy
import os
import pickle

import numpy as np
import rarfile
import torch
from torchvision import datasets

from .data_utils import Partition, get_transform


class EdgeDataset():
    def __init__(self, args, target_label, root="./data"):
        self.args = args
        self.root = root
        self.target_label = target_label
        # Note that the source label of ARDIS is index = 7, and the source label of SouthwestAirline is airplane, index = 0. Therefore, the target label should not be the same with that.
        if self.args.dataset == "CIFAR10":
            self.data_obj = SouthwestAirline(args, self.target_label)
        elif "MNIST" in self.args.dataset:
            self.data_obj = ARDIS(args, self.target_label)
        else:
            raise ValueError(
                f"Unsupported dataset for edgecase: {self.args.dataset}")

    def get_poisoned_trainset(self, sample_len=None):
        x, y = self.data_obj.get_poisoned_trainset()
        if sample_len:
            indices = np.random.choice(
                range(len(x)), sample_len, replace=False)
            x, y = x[indices], y[indices]
        return x, y

    def get_poisoned_testset(self, sample_len=None):
        test_dataset = self.data_obj.get_poisoned_testset()
        if sample_len:
            indices = np.random.choice(
                range(len(test_dataset.data)), sample_len, replace=False)
            test_dataset.data, test_dataset.targets = test_dataset.data[
                indices], test_dataset.targets[indices]
        return test_dataset

    def mix_trainset(self, clean_dataset, poisoning_ratio):
        """
        mix the clean train dataset with the backdoored_dataset
        return the mixed dataset
        """
        # 1. define sample length for benign and malicious data.
        # It trys to keep the total number of data in the mixed dataset to be the same as the clean dataset. However, if the poisoning_ratio is too high and total number it too large, there is no enough edge data to meet the poisoning_ratio, so the total number of data in the mixed dataset will be shorten accordingly.
        total_num = len(clean_dataset)
        poison_num = int(total_num * poisoning_ratio)
        # because the total southwest train dataset is 784, and the total ardis train sdataset is 660
        if self.args.dataset == "CIFAR10":
            poison_num = min(poison_num, 784)
        elif "MNIST" in self.args.dataset:
            poison_num = min(poison_num, 660)
        else:
            raise ValueError(
                f"Unsupported dataset for mix: {self.args.dataset}")
        # determine the number of benign data to meet the poisoning_ratio
        benign_num = int(poison_num / poisoning_ratio - poison_num)

        # 2. sample the benign and malicious data
        train_tran, _ = get_transform(self.args)
        sampled_benign_indices = np.random.choice(
            range(total_num), benign_num, replace=False)
        train_dataset = Partition(
            clean_dataset, sampled_benign_indices, transform=train_tran)

        # 3. get the tensor-type sampled poisoned dataset
        poisoned_x, poisoned_y = self.get_poisoned_trainset(
            poison_num)

        # 4. mix the sampled clean dataset with the poisoned dataset.
        # Note that poisoned_trainset.data and poisoned_x is tensor-type images, so torch.cat is used to concatenate them.
        poisoned_trainset = copy.deepcopy(train_dataset)

        if isinstance(poisoned_trainset.data, np.ndarray):
            poisoned_trainset.data = np.concatenate(
                (poisoned_trainset.data, poisoned_x), axis=0)
            poisoned_trainset.targets = np.concatenate(
                (poisoned_trainset.targets, poisoned_y), axis=0)
        elif isinstance(poisoned_trainset.data, torch.Tensor):
            poisoned_trainset.data = torch.cat(
                (poisoned_trainset.data, poisoned_x), dim=0)
            poisoned_trainset.targets = torch.cat(
                (poisoned_trainset.targets, poisoned_y), dim=0)
        return poisoned_trainset


class SouthwestAirline():
    """This is the southwest Airline plane dataset, which will be labled from `airplane` type to `target` type, trunk
    """

    def __init__(self, args, target_label=None, root="./data/southwest"):
        self.root = root
        self.source_label = 0  # airplane index=0 in CIFAR10
        self.args = args
        self.target_label = target_label if target_label else 9
        source_target_check(self.source_label,  self.target_label)
        self.check_integrity()
        self.read_dataset()

    def check_integrity(self):
        """
        If there is no ./saved_datasets, please download it from [southwest dataset link](https://github.com/ksreenivasan/OOD_Federated_Learning/tree/master/saved_datasets)
        load the dataset from the saved file
        """
        url_location = "https://raw.githubusercontent.com/ksreenivasan/OOD_Federated_Learning/master/saved_datasets/"
        self.filenames = ['southwest_images_new_train.pkl',
                          'southwest_images_new_test.pkl']
        all_files_exist = all(os.path.exists(os.path.join(self.root, file))
                              for file in self.filenames)
        if not all_files_exist:
            # mkdir root data
            os.makedirs(self.root, exist_ok=True)
            # download dataset
            try:
                for file in self.filenames:
                    download_link = os.path.join(url_location, file)
                    datasets.utils.download_url(download_link, self.root)
                print("Successfully downloaded the southwest dataset")
            except Exception as e:
                raise Exception(f"Exception: {e}")

    def read_dataset(self):
        # load the dataset from the saved file
        # Note that the labels are plane types, and we will convert them to target types
        with open(os.path.join(self.root, self.filenames[0]), 'rb') as train_f:
            self.southwest_train_images = pickle.load(train_f)

        with open(os.path.join(self.root, self.filenames[1]), 'rb') as test_f:
            self.southwest_test_images = pickle.load(test_f)

    def get_poisoned_trainset(self):
        # train dataset is provided to be tailed to the benign dataset
        self.train_images = self.southwest_train_images
        # transform is done in mix_trainset later
        # 9 is the default target label, trunk in CIAFR10
        self.train_labels = np.array(
            [self.target_label] * len(self.train_images))

        return self.train_images, self.train_labels

    def get_poisoned_testset(self):
        # test dataset is provided to be feeded into test loader
        self.test_images = self.southwest_test_images
        # 9 is the default target label, trunk in CIAFR10
        self.test_labels = np.array(
            [self.target_label] * len(self.test_images))

        test_trans = get_transform(self.args)[1]
        test_dataset = datasets.CIFAR10(
            './data', train=False, download=False, transform=test_trans)
        test_dataset.data, test_dataset.targets = self.test_images, self.test_labels
        return test_dataset


class ARDIS():
    """
    [ARDIS: a Swedish historical handwritten digit dataset | Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-019-04163-3)

    https://ardisdataset.github.io/ARDIS/

    In edge-case backdoor, we take source label: 7 images, and toward target label: 1/target_label"""

    def __init__(self, args, target_label=None, root="./data"):
        self.root = root
        self.source_label = 7
        self.args = args
        self.target_label = target_label if target_label else 1
        source_target_check(self.source_label,  self.target_label)
        self.check_integrity()
        self.read_dataset()

    def check_integrity(self):
        # check the integrity of the dataset, download if not exist
        self.data_path = f'{self.root}/ARDIS/'
        self.filenames = ['ARDIS_train_2828.csv', 'ARDIS_train_labels.csv',
                          'ARDIS_test_2828.csv', 'ARDIS_test_labels.csv']
        all_files_exist = all(os.path.exists(os.path.join(self.data_path, file))
                              for file in self.filenames)

        if not all_files_exist:
            # download dataset
            download_link = 'https://raw.githubusercontent.com/ardisdataset/ARDIS/master/ARDIS_DATASET_IV.rar'
            datasets.utils.download_url(download_link, self.data_path)
            raw_filename = download_link.split('/')[-1]
            # extract rar to csv, which requires the rarfile and unrar package, and system unrar command
            try:
                with rarfile.RarFile(os.path.join(self.data_path, raw_filename)) as rf:
                    rf.extractall(path=self.data_path)
                print("Successfully downloaded the southwest dataset")

            except Exception as e:
                raise Exception(
                    f"Extraction failed: Please install `rarfile` and `unrar` using `conda install rarfile unrar` or `pip install rarfile unrar`. If the issue persists, ensure `unrar` is installed on your system with `sudo apt install unrar` for Linux, `sud brew install unrar` for MacOS.\n Exception: {e}")
            else:  # no Exception, that is, the extraction is successful
                # remove the downloaded package if the extraction is successful
                os.remove(os.path.join(self.data_path, raw_filename))

    def read_dataset(self):
        # load the data from csv's and convert to tensor, becuase MNIST dataset's data and target are tensor-type format
        def load_cvs(idx): return torch.from_numpy(np.loadtxt(
            os.path.join(self.data_path, self.filenames[idx]), dtype='float32'))
        self.train_images, self.train_labels = load_cvs(0), load_cvs(1)
        self.test_images, self.test_labels = load_cvs(2), load_cvs(3)

        # reshape to be [samples][width][height]
        def to_MNIST(x): return x.reshape(x.shape[0], 28, 28)
        self.train_images, self.test_images = to_MNIST(
            self.train_images), to_MNIST(self.test_images)

        # raw labels are one-hot encoded
        # 1. convert one-hot encoded labels to integer labels
        def onehot_to_label(y): return np.argmax(
            y, axis=1)  # default return y-type: tensor
        self.train_labels, self.test_labels = onehot_to_label(
            self.train_labels), onehot_to_label(self.test_labels)

        # 2. get the images and labels for digit 7
        def filter(train_x, train_y):
            # Note that if use argwhere the 1-dimension will added, which is not expected
            indices = train_y[train_y == self.source_label]
            sampled_images = train_x[indices]
            sampled_labels = torch.tensor([self.source_label] * len(indices))
            return sampled_images, sampled_labels

        # sample source label=7 train images as sampled_train_images. test images are fully used wihout sampling.
        self.sampled_train_images, self.sampled_train_labels = filter(
            self.train_images, self.train_labels)
        self.sampled_test_images, self.sampled_test_labels = self.test_images, self.test_labels

    def get_poisoned_trainset(self):
        # train_trans = get_transform(self.args)[0]
        # train_dataset = datasets.MNIST('./data', train=True, download=True,
        #                                transform=train_trans)
        # transform is done in mix_trainset later
        self.posioned_labels = torch.tensor(
            [self.target_label] * len(self.sampled_train_labels))
        # train_dataset.data, train_dataset.targets = self.images, self.labels
        # return train_dataset
        return self.sampled_train_images, self.posioned_labels

    def get_poisoned_testset(self):
        # args.dataset
        test_trans = get_transform(self.args)[1]
        test_dataset = datasets.MNIST(
            './data', train=False, download=False, transform=test_trans)
        self.posioned_labels = torch.tensor(
            [self.target_label] * len(self.sampled_test_labels))
        test_dataset.data, test_dataset.targets = self.sampled_test_images, self.posioned_labels
        return test_dataset


def source_target_check(source_label, target_label):
    if source_label == target_label:
        raise ValueError(
            f"The source label and target label should not be the same, currently they are both {source_label}")


if __name__ == "__main__":
    class args:
        dataset = "CIFAR10"
    tmp = EdgeDataset(args)
