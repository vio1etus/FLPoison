import torch
import torch.nn as nn
import torch.nn.functional as F
from fl.models import model_registry

simple_config = {"28": [30, 50, 100], # for MNIST, FashionMNIST, FEMNIST
"32": [32, 64, 512] # for CIFAR-10, CINIC-10
}

@model_registry
class simplecnn(nn.Module):
    """
    Widely used simple CNN architecture for image classification.
    References: FLTrust, FLdetector, FangAttack,...
    """
    def __init__(self, input_size=(3, 32, 32), num_classes=10):
        super().__init__()
        self.model_config = simple_config[f"{input_size[1]}"]
        # 定义卷积层 + ReLU
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=self.model_config[0], kernel_size=3)  # 输入通道为3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=self.model_config[0], out_channels=self.model_config[1], kernel_size=3)  # 输入通道为30
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算经过卷积和池化后的特征图大小
        self.flattened_size = self._get_flattened_size(input_size)

        # 定义全连接层
        self.fc1 = nn.Linear(self.flattened_size, self.model_config[2])  # 全连接层
        self.fc2 = nn.Linear(self.model_config[2], num_classes)  # 输出层

    def _get_flattened_size(self, input_size):
        # 输入尺寸
        height, width = input_size[1], input_size[2]

        # 卷积层 1
        height = (height - 3 + 0) // 1 + 1  # kernel_size=3, padding=0, stride=1
        width = (width - 3 + 0) // 1 + 1
        
        # 池化层 1
        height = (height - 2) // 2 + 1  # pool_size=2, stride=2
        width = (width - 2) // 2 + 1
        
        # 卷积层 2
        height = (height - 3 + 0) // 1 + 1  # kernel_size=3, padding=0, stride=1
        width = (width - 3 + 0) // 1 + 1
        
        # 池化层 2
        height = (height - 2) // 2 + 1  # pool_size=2, stride=2
        width = (width - 2) // 2 + 1
        
        return self.model_config[1] * height * width  # 50是第二个卷积层的输出通道数

    def forward(self, x):
        # 前向传播
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 将特征图展平
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 最后一层不使用激活函数（softmax将在损失函数中处理）

        return x

if __name__ == "__main__":
    model = simplecnn(input_size=(3, 32, 32), num_classes=10)
    print(model)
    # 假设输入是一个批次的图像，形状为 [batch_size, channels, height, width]
    sample_input = torch.randn(1, 3, 32, 32)  # 1 个样本，3 个通道，32x32 像素
    output = model(sample_input)
    print(output.shape)