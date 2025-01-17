import torch.nn as nn
import torch.nn.functional as F
from fl.models import model_registry


@model_registry
class lenet(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


@model_registry
class lenet_bn(nn.Module):
    @staticmethod
    def weight_init(m):
        # 1. According to the different network layers, define different initialization methods
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # check conv2d, init with kaiming_normal
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
        # check batchnorm
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.convnet = nn.Sequential(
            # in_channels is the number of color channels of the input image
            # CIFAR-10 three color channels
            nn.Conv2d(in_channels=num_channels, out_channels=6,
                      kernel_size=5),  # MNIST 1 color channel
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.LogSoftmax(dim=-1))
        self.apply(self.weight_init)

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
