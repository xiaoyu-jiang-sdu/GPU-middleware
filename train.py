import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet18
from wrapper.wrapper import ONNXModelWrapper

# 原始模型
base_model = resnet18(num_classes=10)

# 包装
model = ONNXModelWrapper(
    base_model,
    input_shape=(1, 3, 224, 224),
    backend="cuda"
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(base_model.parameters(), lr=0.01)

# 假数据
x = torch.randn(4, 3, 224, 224)
y = torch.randint(0, 10, (4,)).to(torch.device("cuda"))

out = model(x)
loss = criterion(out, y)

loss.backward()
optimizer.step()

print("OK, backward & optimizer work.")
