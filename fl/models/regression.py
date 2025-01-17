
import torch.nn as nn
from fl.models import model_registry


@model_registry
class lr(nn.Module):
    """logistic regression model"""
    def __init__(self, input_dim=32*32, num_classes=10):
        super().__init__()
        self.input_dim, self.num_classes = input_dim, num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, xb):
        # flatten the image to vector for linear
        xb = xb.reshape(-1, self.input_dim)
        outputs = self.linear(xb)
        return outputs
