"""
YOLOv1 Backbone - VGG16 feature extractor

VGG16 conv layers extract features, followed by an extra conv layer
to produce a 7x7 feature map with 1024 channels for the detection head.

Input:  (N, 3, 448, 448)
Output: (N, 1024, 7, 7)
"""

import torch
import torch.nn as nn
from torchvision import models


class VGG16Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        vgg16 = models.vgg16(pretrained=pretrained)
        #  448 -> 224 -> 112 -> 56 -> 28 -> 14
        self.conv_layers = vgg16.features  # output: (N, 512, 14, 14)

        self.extra = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # (N, 512, 14, 14)
        x = self.extra(x)        # (N, 1024, 7, 7)
        return x
