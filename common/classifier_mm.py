"""classifier_mm.py

Multi-modal preictal-baseline classifier. Mirrors ``common.classifier.EEGClassifier``
but accepts arbitrary input-channel counts so we can train preictal classifiers
on (EEG | EEG+EMG | EEG+Photo | EEG+EMG+Photo) spectrogram stacks.

This file is intentionally separate from ``common/classifier.py`` so the
production diffusion pipeline keeps its frozen 1-channel target classifier
unchanged. Importing this module has zero side effects on the rest of the
codebase.

Pretrained-weight init strategy (when ``pretrained=True``):
  - For ``in_channels == 1``, behave like ``EEGClassifier``: average the
    three RGB filters into a single channel (matching the existing v6
    pipeline so cross-comparisons are apples-to-apples).
  - For ``in_channels > 1``, copy the RGB-averaged 1-channel kernel into
    every input channel and divide by ``in_channels`` so the activation
    magnitude is preserved when a uniformly-zero-mean spectrogram is fed in.

The classifier head matches ``EEGClassifier`` exactly so summary.json files
written by ``train_classifier_mm.py`` can be read by the existing aggregator
without schema changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MMClassifier(nn.Module):
    """ResNet18 modified for ``C_in``-channel 224x224 spectrogram input."""

    def __init__(
        self,
        n_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        self.in_channels = int(in_channels)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                # First reduce ImageNet's 3-channel RGB conv to a single-channel
                # kernel by mean-pooling across RGB, then tile across our
                # `in_channels` and rescale so input magnitude is preserved.
                gray = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                tiled = gray.repeat(1, self.in_channels, 1, 1) / float(self.in_channels)
                self.conv1.weight.data = tiled

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
        """Input [B, C_in, 224, 224] -> logits [B, n_classes]."""
        feat = self.encode(x)
        return self.classifier(feat)


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
