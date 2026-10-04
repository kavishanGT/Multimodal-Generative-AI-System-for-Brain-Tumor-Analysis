# models/deit_model.py
# DeiT-Small (Vision Transformer) for Brain Tumor Classification

import torch
import torch.nn as nn
from transformers import DeiTModel


class BrainTumorDeiTSmall(nn.Module):
    """
    DeiT-Small pretrained on ImageNet (upgraded from DeiT-Tiny).
    Fine-tunes last 2 encoder layers + custom classifier head.
    ~22M parameters, attention mechanism captures global context.
    """

    def __init__(self, num_classes=4):
        super(BrainTumorDeiTSmall, self).__init__()

        # Load pretrained DeiT-Small (upgraded from Tiny)
        self.vit = DeiTModel.from_pretrained(
            "facebook/deit-small-patch16-224"
        )

        # Freeze all layers first
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last 2 encoder layers for fine-tuning
        for layer in self.vit.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # DeiT-Small hidden size = 384 (vs Tiny = 192)
        hidden_size = self.vit.config.hidden_size  # 384

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits
