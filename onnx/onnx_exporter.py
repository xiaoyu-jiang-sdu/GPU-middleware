import torch
import torchvision.models as models

# 1️⃣ 创建 ResNet-50 模型（预训练或者随机权重）
model = models.resnet50(weights=None)  # 若想使用预训练：weights=models.ResNet50_Weights.DEFAULT
model.eval()  # 导出模型前切换到评估模式

# 2️⃣ 构造输入张量
# ResNet-50 输入是 [batch_size, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224)

# 3️⃣ 导出 ONNX
torch.onnx.export(
    model,                   # PyTorch 模型
    dummy_input,             # 输入张量
    "resnet50.onnx",         # 输出文件
    opset_version=12,        # ONNX 版本，推荐 >= 12
    input_names=["input"],   # 输入节点名称
    output_names=["output"], # 输出节点名称
    dynamic_axes={           # 可选：支持动态 batch_size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ResNet-50 已导出为 resnet50.onnx")