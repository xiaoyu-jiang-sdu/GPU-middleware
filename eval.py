import torch

from torchvision.models import resnet18
from model.model import ONNXAwareModel

# 原始模型
base_model = resnet18(num_classes=10)
base_model.eval()

# 包装
model = ONNXAwareModel(
    base_model,
    input_shape=(1, 3, 224, 224),
    backend="cuda"
)
# 假数据
x = torch.randn(1, 3, 224, 224)

out = model(x)

print("eval finish! out:", out)