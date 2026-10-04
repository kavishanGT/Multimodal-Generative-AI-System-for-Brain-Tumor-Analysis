# models/efficientnet_model.py
# EfficientNet-B0 for Brain Tumor Classification

import torch
import torch.nn as nn
import timm


class BrainTumorEfficientNet(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet.
    Fine-tunes the last 2 blocks + custom classifier head.
    ~5.3M parameters, excellent efficiency/accuracy tradeoff.
    """

    def __init__(self, num_classes=4):
        super(BrainTumorEfficientNet, self).__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0  # Remove original classifier
        )

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 blocks for fine-tuning
        for name, param in self.backbone.named_parameters():
            if "blocks.5" in name or "blocks.6" in name or "conv_head" in name or "bn2" in name:
                param.requires_grad = True

        # Get feature dimension
        feature_dim = self.backbone.num_features  # 1280 for B0

        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
