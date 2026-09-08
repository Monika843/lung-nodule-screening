
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = densenet121(weights=weights)
    except Exception:
        # Falls back to random init if ImageNet weights can't be downloaded
        # (e.g. restricted/offline network). On your own machine or Colab
        # with normal internet access, pretrained=True will download fine.
        print("Warning: could not download pretrained ImageNet weights, "
              "using random initialization instead.")
        model = densenet121(weights=None)
        pretrained = False

    # CT slices are single-channel; ImageNet DenseNet121 expects 3.
    # Average the pretrained first-conv weights across channels so we keep
    # transfer-learned low-level filters instead of random-initializing.
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.features.conv0 = new_conv

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
