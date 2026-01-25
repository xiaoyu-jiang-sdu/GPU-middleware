from torchvision import models
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, ResNet18_Weights

from models.registry import register_model, infer_shape


@register_model('resnet18')
def build_resnet18(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 从 weights 推断输入尺寸
    input_shape = infer_shape(weights)

    return model, input_shape


@register_model("resnet50")
def build_resnet50(num_classes: int):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    input_shape = infer_shape(weights)

    return model, input_shape


@register_model("vit_b_16")
def build_vit_b16(num_classes: int):
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

    if num_classes != 1000:
        model.heads.head = nn.Linear(
            model.heads.head.in_features, num_classes
        )

    input_shape = infer_shape(weights)

    return model, input_shape
