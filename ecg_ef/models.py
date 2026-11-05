"""Model zoo: a custom ECG CNN plus standard ImageNet architectures from timm."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Channel attention: lets the network weight the feature maps (e.g. ST/T-wave or
    QRS-morphology detectors) that matter most for each ECG."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // r, 1),
                                nn.ReLU(inplace=True), nn.Conv2d(ch // r, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ResSEBlock(nn.Module):
    """Residual block with time-elongated (3x5) kernels + squeeze-excitation.
    ECG features (P/QRS/T morphology, intervals) extend mostly along the time
    (horizontal) axis, so the kernels see more time than amplitude."""

    def __init__(self, cin: int, cout: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, (3, 5), stride, (1, 2), bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (3, 5), 1, (1, 2), bias=False), nn.BatchNorm2d(cout),
            SqueezeExcite(cout),
        )
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.drop(self.body(x)) + self.skip(x))


class ECGNet(nn.Module):
    """Residual SE-CNN for 12-lead ECG images (~6.3M parameters).

    1x256x512 -> stem 7x7/2 -> 4 stages (stride 2 each) -> [GAP || GMP] -> FC(4)
    Global max-pooling keeps strong local evidence (e.g. a single abnormal QRS in one
    lead) that average pooling alone would dilute.
    """

    def __init__(self, num_classes: int = 4, in_chans: int = 1,
                 widths=(32, 64, 128, 192, 256), depths=(1, 1, 2, 2), drop: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, widths[0], 7, 2, 3, bias=False), nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (cout, d) in enumerate(zip(widths[1:], depths)):
            for j in range(d):
                layers.append(ResSEBlock(cin, cout, stride=2 if j == 0 else 1,
                                         drop=0.1 if i >= 2 else 0.0))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.num_features = cin
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * cin, num_classes))

    def forward_features(self, x):
        return self.features(self.stem(x))

    def forward_head(self, f):
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        return self.head(f)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


TIMM_MODELS = {
    "vgg16": "vgg16",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
    "mobilenetv2": "mobilenetv2_100",
    "mobilenetv3": "mobilenetv3_large_100",
    "inception_v3": "inception_v3",
    "convnext_atto": "convnext_atto",
}
MODEL_NAMES = ["ecgnet", *TIMM_MODELS]


def build_model(name: str, num_classes: int = 4, pretrained: bool = False,
                drop_rate: float = 0.3) -> nn.Module:
    if name == "ecgnet":
        return ECGNet(num_classes)
    if name not in TIMM_MODELS:
        raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
    # in_chans=1: timm sums the pretrained RGB filters into one grayscale filter.
    return timm.create_model(TIMM_MODELS[name], pretrained=pretrained, in_chans=1,
                             num_classes=num_classes, drop_rate=drop_rate)
