import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_mobilenetv2(num_classes=8, pretrained=True, freeze=False):
    """
    Returns a MobileNetV2 model, with an option to replace the final layer
    and freeze the feature extractor.
    
    :param num_classes: Number of output classes (e.g., 8 for classification).
    :param pretrained: Whether to initialize with ImageNet-pretrained weights.
    :param freeze: Whether to freeze all feature-extraction layers (only train final layers).
    :return: A MobileNetV2 model ready for training/fine-tuning.
    """
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1  # or .DEFAULT for latest
    else:
        weights = None
    
    model = mobilenet_v2(weights=weights)

    # Replace the final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Freeze feature extraction layers if desired
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    return model

if __name__ == "__main__":
    # Example usage
    model = get_mobilenetv2(num_classes=8, pretrained=True, freeze=False)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"✅ Trainable parameters: {sum(p.numel() for p in trainable_params)}")