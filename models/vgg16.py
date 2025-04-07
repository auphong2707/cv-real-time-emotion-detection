import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def get_vgg16(num_classes=7, pretrained=True, freeze=False):
    """
    Returns a VGG16 model with a custom classifier and optional freezing.

    :param num_classes: Number of output classes (e.g., 7 for FER).
    :param pretrained: Whether to use ImageNet pretrained weights.
    :param freeze: Whether to freeze the convolutional feature extractor.
    :return: A VGG16 model ready for training/fine-tuning.
    """
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)

    # Replace the classifier
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    # Optionally freeze the convolutional base
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    return model

# Optional test block
if __name__ == "__main__":
    model = get_vgg16(num_classes=8, pretrained=True, freeze=False)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"✅ Trainable parameters: {sum(p.numel() for p in trainable_params)}")
