import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_efficientnet_b0(num_classes=8, pretrained=True, freeze=False):
    """
    Returns an EfficientNet-B0 model with a custom classifier and optional freezing.

    :param num_classes: Number of output classes (e.g., 7 for FER).
    :param pretrained: Whether to use ImageNet pretrained weights.
    :param freeze: Whether to freeze the convolutional feature extractor.
    :return: A modified EfficientNet-B0 model ready for training/fine-tuning.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    # Optionally freeze the feature extractor
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    return model

# Optional test block
if __name__ == "__main__":
    model = get_efficientnet_b0(num_classes=8, pretrained=True, freeze=False)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"✅ Trainable parameters: {sum(p.numel() for p in trainable_params)}")
