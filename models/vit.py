import timm
import torch.nn as nn

def get_vit(num_classes=8, pretrained=True):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

    # Replace the classification head
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
