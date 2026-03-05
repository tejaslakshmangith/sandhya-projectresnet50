import torch
import torch.nn as nn
from torchvision import models


class SmartMineResNet50(nn.Module):
    """
    ResNet-50 transfer learning model for SmartMine image classification.
    Freezes convolutional backbone, replaces FC head for custom classes.
    """

    def __init__(self, num_classes: int):
        super(SmartMineResNet50, self).__init__()

        # Load pretrained ResNet-50
        self.model = models.resnet50(pretrained=True)

        # Freeze all backbone layers for transfer learning
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
