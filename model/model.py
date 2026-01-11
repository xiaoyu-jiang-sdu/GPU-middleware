import torch.nn as nn

from utils.onnx_utils import export_onnx, parse_onnx_model
from IR.executor import IRExecutor
from adapter.factory import create_adapter


class ONNXAwareModel(nn.Module):
    def __init__(self, model: nn.Module, input_shape, backend="cuda"):
        """
        model: 原始 torch.nn.Module（不修改）
        input_shape: 用于 ONNX export
        backend: cuda / mlu / npu
        """
        super().__init__()
        self.model = model
        self.backend = backend

        # 导出 ONNX
        onnx_model = export_onnx(model, input_shape)

        # 构建 IR
        self.ir_graph = parse_onnx_model(onnx_model)

        # 创建 adapter + executor
        self.adapter = create_adapter(backend)
        self.executor = IRExecutor(self.ir_graph, self.adapter)

    def forward(self, x):
        return self.executor.run(
            inputs={"input": x},
            parameters=dict(self.model.named_parameters())
        )
