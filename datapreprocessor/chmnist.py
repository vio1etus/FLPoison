import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets.utils import download_and_extract_archive

class CHMNIST(Dataset):
    dataset_url = "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1"
    zip_md5 = "0ddbebfc56344752028fda72602aaade"
    def __init__(self, root, train=True, download=False, transform=None, test_split=0.2, random_seed=42):
        """
        Args:
            root (str): Root directory of the dataset folder, Kather_texture_2016_image_tiles_5000, where each subdirectory corresponds to a class.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load the training set or the test set.
            test_split (float): Proportion of the dataset to use as the test set.
            random_seed (int): Random seed for reproducibility of the train-test split.
        """
        self.root_dir = os.path.join(root, "Kather_texture_2016_image_tiles_5000")
        self.transform = transform
        self.train = train

        # Automatically download and extract dataset if not present
        if download:
            if not os.path.exists(self.root_dir):
                print("Dataset not found. Downloading...")
                download_and_extract_archive(
                    url=self.dataset_url,
                    download_root=root,
                    extract_root=root,
                    filename="chmnist.zip",
                    md5=self.zip_md5,
                    remove_finished=True
                )

        self.classes = sorted(os.listdir(self.root_dir))  # Assumes subdirectories are class names
        self.image_paths = []
        self.targets = []

        # Collect all image paths and their corresponding labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.targets.append(label)

        # Split into train and test sets
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.targets, test_size=test_split, random_state=random_seed, stratify=self.targets
        )

        if self.train:
            self.image_paths = train_paths
            self.targets = train_labels
        else:
            self.image_paths = test_paths
            self.targets = test_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the class index.
        """
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 8
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet requires 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader
    dataset_root = "data/Kather_texture_2016_image_tiles_5000"
    
    train_dataset = CHMNIST(root_dir=dataset_root, transform=train_transform, train=True, test_split=0.2)
    test_dataset = CHMNIST(root_dir=dataset_root, transform=test_transform, train=False, test_split=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Class names: {train_dataset.classes}")
    
    # Model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)