import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datapreprocessor.cinic10 import CINIC10
from plot_utils import plot_label_distribution


def load_data(args):
    # load dataset
    trans, test_trans = get_transform(args)
    data_directory = './data'
    if args.dataset == "EMNIST":
        train_dataset = datasets.EMNIST(data_directory, split="digits", train=True, download=True,
                                        transform=trans)
        test_dataset = datasets.EMNIST(
            data_directory, split="digits", train=False, transform=test_trans)
    elif args.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        train_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=True,
                                                         download=True, transform=trans)
        test_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=False,
                                                        download=True, transform=test_trans)
    elif args.dataset == "CINIC10":
        train_dataset = CINIC10(root=data_directory, train=True, download=True,
                                transform=trans)
        test_dataset = CINIC10(root=data_directory, train=True, download=True,
                               transform=test_trans)
    else:
        raise ValueError("Dataset not implemented yet")

    # deal with CIFAR10 list-type targets. CIFAR10 data is numpy array defaultly.
    train_dataset.targets = list_to_tensor(train_dataset.targets)
    test_dataset.targets = list_to_tensor(test_dataset.targets)
    return train_dataset, test_dataset


def list_to_tensor(vector):
    """
    check whether a instance is tensor, convert it to tensor if it is a list.
    """
    if isinstance(vector, list):
        vector = torch.tensor(vector)
    return vector


def subset_by_idx(args, dataset, indices, train=True):
    trans = get_transform(args)[0] if train else get_transform(args)[1]
    dataset = Partition(
        dataset, indices, transform=trans)
    return dataset


def get_transform(args):
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST", "FEMNIST"] and args.model in ['lenet', "lr"]:
        # resize MNIST to 32x32 for LeNet5
        train_tran = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
        test_trans = train_tran
        # define the image dimensions for self.args, so that others can use it, such as DeepSight, lr model
        args.num_dims = 32
    elif args.dataset in ["CINIC10"]:
        train_tran = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)])
        test_trans = train_tran
    elif args.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        args.num_dims = 32 if args.dataset in ['CIFAR10', 'CIFAR100'] else 64
        # data augmentation
        train_tran = transforms.Compose([
            # transforms.RandomCrop(args.num_dims, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
    else:
        raise ValueError("Dataset not implemented yet")

    return train_tran, test_trans


def split_dataset(args, train_dataset, test_dataset):
    # agrs.cache_partition: True, False, non-iid, iid, class-imbalanced-iid
    cache_flag = (args.cache_partition ==
                  True or args.cache_partition == args.distribution)
    if cache_flag:
        # ready for cache usage
        # check if the indices are already generated in running_caches folder
        cache_exist, file_path = check_partition_cache(args)
        if cache_exist:
            args.logger.info("Target indices caches to save time")
            with open(file_path, 'rb') as f:
                client_indices = pickle.load(f)
            return client_indices, test_dataset

    args.logger.info("Generating new indices")
    if args.distribution in ['iid', 'class-imbalanced_iid']:
        client_indices = iid_partition(args, train_dataset)
        args.logger.info("Doing iid partition")
        if "class-imbalanced" in args.distribution:
            args.logger.info("Doing class-imbalanced iid partition")
            # class-imbalanced iid partition
            for i in range(args.num_clients):
                class_indices = client_class_indices(
                    client_indices[i], train_dataset)
                client_indices[i] = class_imbalanced_partition(
                    class_indices, args.im_iid_gamma)
    elif args.distribution in ['non-iid']:
        # dirichlet partition
        args.logger.info("Doing non-iid partition")
        client_indices = dirichlet_split_noniid(
            train_dataset.targets, args.dirichlet_alpha, args.num_clients)
        args.logger.info(f"dirichlet alpha: {args.dirichlet_alpha}")
    if cache_flag:
        save_partition_cache(client_indices, file_path)
        # if "class-imbalanced" in args.distribution:
        #     # class-imbalanced partition for test dataset for evaluation
        #     test_class_indices = dataset_class_indices(test_dataset)
        #     test_class_indices = class_imbalanced_partition(
        #         test_class_indices, args.im_iid_gamma)
        #     test_dataset = subset_by_idx(
        #         args, test_dataset, test_class_indices)
    args.logger.info(f"{args.distribution} partition finished")
    # plot the visualization of label distribution of the clients
    # plot_label_distribution(train_dataset, client_indices, args.num_clients, args.dataset, args.distribution)
    return client_indices, test_dataset


def save_partition_cache(client_indices, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(client_indices, f)


def check_partition_cache(args):
    cache_exist = None
    folder_path = 'running_caches'
    file_name = f'{args.dataset}_balanced_{args.distribution}_{args.num_clients}_indices'
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        cache_exist = True if os.path.exists(file_path) else False
    return cache_exist, file_path


def check_noniid_labels(args, train_dataset, client_indices):
    """
    check the unique labels of each client and the common labels across all clients
    """
    client_unique_labels = {}
    common_labels = None
    for client_id, indices in enumerate(client_indices):
        # get the labels of the corresponding indices
        labels = train_dataset.targets[indices]
        # get the unique labels of the client
        unique_labels = set(labels.tolist())
        client_unique_labels[client_id] = unique_labels
        # for the first client, initialize common_labels as the unique labels
        if common_labels is None:
            common_labels = unique_labels
        else:
            # update common_labels by taking the intersection of the unique labels
            common_labels = common_labels.intersection(unique_labels)

    # log the unique labels of each client and the common labels across all clients
    args.logger.info(
        f"Common unique labels across all clients: {common_labels}")
    for client_id, unique_labels in client_unique_labels.items():
        args.logger.info(
            f"Client {client_id} has unique labels: {unique_labels}")


class Partition(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.classes = dataset.classes
        self.indices = indices if indices is not None else range(len(dataset))
        self.data, self.targets = dataset.data[self.indices], dataset.targets[self.indices]
        # (N, C, H, W) or (N, H, W) for MNIST-like grey images, mode='L'; CIFAR10-like color images, mode='RGB'
        self.mode = 'L' if len(self.data.shape) == 3 else 'RGB'
        self.transform = transform
        self.poison = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # convert image to numpy array. for MNIST-like dataset, image is torch tensor, for CIFAR10-like dataset, image type is numpy array.
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy()
        # to return a PIL Image
        image = Image.fromarray(image, mode=self.mode)
        if self.transform:
            image = self.transform(image)

        if self.poison:
            image, target = self.synthesizer.backdoor_batch(
                image, target.reshape(-1, 1))
        return image, target.squeeze()

    def poison_setup(self, synthesizer):
        self.poison = True
        self.synthesizer = synthesizer


def iid_partition(args, train_dataset):
    """
    nearly-quantity-balanced and class-balanced IID partition for clients.
    """
    labels = train_dataset.targets
    client_indices = [[] for _ in range(args.num_clients)]
    for cls in range(len(train_dataset.classes)):
        # get the indices of current class
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # get the number of sample class=cls indices for each client
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # random permutation
        class_indices = class_indices[torch.randperm(len(class_indices))]

        # calculate the number of samples for each client
        num_samples = len(class_indices)
        num_samples_per_client_per_class = num_samples // args.num_clients
        # other remaining samples
        remainder_samples = num_samples % args.num_clients

        # uniformly distribute the samples to each client
        for client_id in range(args.num_clients):
            start_idx = client_id * num_samples_per_client_per_class
            end_idx = start_idx + num_samples_per_client_per_class
            client_indices[client_id].extend(
                class_indices[start_idx:end_idx].tolist())
        # distribute the remaining samples to the first few clients
        for i in range(remainder_samples):
            client_indices[i].append(
                class_indices[-(i + 1)].item())
    client_indices = [torch.tensor(indices) for indices in client_indices]
    return client_indices


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Function: divide the sample index set into n_clients subsets according to the Dirichlet distribution with parameter alpha
    References:
    [orion-orion/FedAO: A toolbox for federated learning](https://github.com/orion-orion/FedAO)
    [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)
    '''
    n_classes = train_labels.max()+1
    # (K, N) category label distribution matrix X, recording the proportion of each category assigned to each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) records the sample index set corresponding to K classes
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # Record the sample index sets corresponding to N clients
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split divides the sample index k_idcs of class k into N subsets according to the proportion fracs
        # i represents the i-th client, idcs represents its corresponding sample index set
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def dataset_class_indices(dataset, class_label=None):
    num_classes = len(dataset.classes)
    if class_label:
        return torch.tensor(np.where(dataset.targets == class_label)[0])
    else:
        class_indices = [torch.tensor(np.where(dataset.targets == i)[
            0]) for i in range(num_classes)]
        return class_indices


def client_class_indices(client_indice, train_dataset):
    """
    Given the a client indice, return the list of indices of each class
    """
    labels = train_dataset.targets
    return [client_indice[labels[client_indice] == cls] for cls in range(len(train_dataset.classes))]


def class_imbalanced_partition(class_indices, im_iid_gamma, method='exponential'):
    """
    Perform exponential sampling on the number of each classes.

    Args:
        class_indices (list): A list of tensor containing index of each class for each client
        gamma (float): The exponential decay rate (0 < gamma <= 1).
        method (str, optional): The sampling method, exponential or step. Default as 'exponential'.

    Returns:
        sampled_class_indices (1d tensor): exponential-sampled class_indices
    """
    num_classes = len(class_indices)
    num_sample_per_class = [max(1, int(im_iid_gamma**(i / (num_classes-1)) * len(class_indices[i])))
                            for i in range(num_classes)]
    sampled_class_indices = [class_indices[i][torch.randperm(
        len(class_indices[i]))[:num_sample_per_class[i]]] for i in range(num_classes)]
    # print(f"num_sample_per_class: {num_sample_per_class}")
    return torch.cat(sampled_class_indices)


if __name__ == "__main__":
    pass
