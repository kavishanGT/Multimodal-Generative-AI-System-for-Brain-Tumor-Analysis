# models/resnet_model.py
# ResNet-50 for Brain Tumor Classification

import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorResNet50(nn.Module):
    """
    ResNet-50 pretrained on ImageNet.
    Fine-tunes layer4 + custom classifier head.
    ~25.6M parameters, gold standard for medical imaging.
    """

    def __init__(self, num_classes=4):
        super(BrainTumorResNet50, self).__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 for fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Get feature dimension before FC
        feature_dim = self.backbone.fc.in_features  # 2048

        # Replace the original FC with our classifier
        self.backbone.fc = nn.Identity()

        # Custom classification head
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
