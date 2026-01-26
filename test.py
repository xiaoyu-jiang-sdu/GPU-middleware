from models.registry import build_model
from wrapper.wrapper import ONNXModelWrapper
import os

os.environ["TORCH_HOME"] = "E:/torch_cache"

base_model, input_shape = build_model(
    model_name="resnet18",
    num_classes=10,
)

model = ONNXModelWrapper(base_model, input_shape)
print(model.op_adapter_mapping())
