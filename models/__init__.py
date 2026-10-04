# models/__init__.py

from models.efficientnet_model import BrainTumorEfficientNet
from models.resnet_model import BrainTumorResNet50
from models.deit_model import BrainTumorDeiTSmall

MODEL_REGISTRY = {
    "EfficientNet-B0": BrainTumorEfficientNet,
    "ResNet-50": BrainTumorResNet50,
    "DeiT-Small": BrainTumorDeiTSmall,
}
