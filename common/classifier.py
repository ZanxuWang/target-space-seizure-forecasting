"""classifier.py

Single-channel ResNet18 classifier for 3s EEG spectrograms.

This is a simplified, EEG-only version of the multi-modal cross-attention
classifier in the original research notebook. It serves two roles in this
pipeline:

1. Horizon-baseline classifier: trained on real preictal spectrograms from a
   given horizon to report how far back seizure-discriminative signal extends.
2. Target auxiliary classifier: trained on real 0-3s post-onset spectrograms,
   then frozen and used to supply the auxiliary CE loss during diffusion
   training, and to evaluate generated 0-3s spectrograms at test time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EEGClassifier(nn.Module):
    """ResNet18 modified for single-channel 224x224 spectrogram input."""

    def __init__(self, n_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Replace the first 3-channel conv with a 1-channel conv, initialized
        # from the mean of the RGB weights if pretrained.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.feat_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d feature vector for the input batch."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input [B, 1, 224, 224] -> logits [B, n_classes]."""
        feat = self.encode(x)
        return self.classifier(feat)


def freeze(model: nn.Module) -> nn.Module:
    """Set model to eval mode and disable gradients on all parameters."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
