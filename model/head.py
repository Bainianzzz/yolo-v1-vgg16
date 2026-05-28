"""
YOLOv1 Detection Head

Takes the 7x7x1024 feature map from backbone and outputs SxSx(B*5+C)
predictions. Uses fully-connected layers as in the original YOLOv1 paper.

Input:  (N, 1024, 7, 7)
Output: (N, 7, 7, B*5 + C)  where B=2, C=80  ->  (N, 7, 7, 90)
"""

import torch
import torch.nn as nn


class YOLOv1Head(nn.Module):
    def __init__(self, grid_size: int = 7, num_bboxes: int = 2, num_classes: int = 80):
        super().__init__()

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        in_features = self.S * self.S * 1024
        out_features = self.S * self.S * (self.B * 5 + self.C)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1024, 7, 7)
        x = self.fc(x)  # (N, S*S*(B*5+C))
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)
