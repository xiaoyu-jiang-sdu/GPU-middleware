import torch
from torchvision.models import resnet18
from model.model import ONNXAwareModel
import numpy as np


seed = 2026
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 原始模型
base_model = resnet18(num_classes=10)

# 包装
model = ONNXAwareModel(
    base_model,
    input_shape=(1, 3, 224, 224),
    backend="cuda"
)
# 随机数据
x = torch.randn(1, 3, 224, 224)

out = model(x)

np_out = out.cpu().numpy()
# np_out = out
# np_out = out.to_numpy()
print("eval finish! out info:")
print(np_out)
print("shape:", np_out.shape)
print("mean:", np_out.mean(), "max:", np_out.max(), "min:", np_out.min())