from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
import hashlib

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

class TinyImageNet(Dataset):
    """
    TinyImageNet Dataset wrapped to resemble PyTorch's built-in datasets like MNIST.
    """
    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ['train', 'val'] # test folder has no labels, so not included

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        Args:
            root (str): Root directory where the dataset will be stored.
            train (bool): True or False. Specifies the dataset split (train or val).
            transform (callable, optional): A function/transform to apply to the images.
            download (bool): If True, downloads the dataset if not already present.
        """
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.data = []
        self.targets = []
        split = 'train' if train else 'val'
        self._load_metadata(split)
    
    def _path(self, *paths):
        """Helper function to construct paths."""
        return os.path.join(self.root, self.base_folder, *paths)
    
    def _load_metadata(self, split):
        data_dir = os.path.join(self.root, self.base_folder)
        wnid_to_label = {}
        with open(self._path('wnids.txt')) as f:
            wnids = [line.strip() for line in f.readlines()]
            wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

        if split == 'train':
            for wnid in wnids:
                imgs_dir = self._path('train', wnid, 'images')
                for img_name in os.listdir(imgs_dir):
                    self.data.append(os.path.join(imgs_dir, img_name))
                    self.targets.append(wnid_to_label[wnid])
        else:
            val_img_dir = self._path('val', 'images')
            val_labels_path = os.path.join(data_dir, 'val', 'val_annotations.txt')
            val_dict = {}
            with open(val_labels_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    val_dict[parts[0]] = wnid_to_label[parts[1]]
            for img_name in os.listdir(val_img_dir):
                if img_name in val_dict:
                    self.data.append(os.path.join(val_img_dir, img_name))
                    self.targets.append(val_dict[img_name])
        self.classes = np.unique(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def download(self):
        if os.path.exists(self._path('train')) and os.path.exists(self._path('val')):
            return
        else:
            print("Dataset folder incomplete. Check zip file...")
            # Check if the zip file exists to verify MD5
            zip_path = os.path.join(self.root, self.filename)
            if os.path.exists(zip_path):
                # Calculate MD5 of the existing file
                md5_hash = hashlib.md5()
                with open(zip_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        md5_hash.update(byte_block)
                md5_actual = md5_hash.hexdigest()
                
                if md5_actual == self.md5:
                    print("Zip file exists and is complete. Unzipping...")
                    extract_archive(zip_path, self.root)
                else:
                    print("Dataset exists but is corrupted. Re-downloading...")
                    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
            else:
                download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)