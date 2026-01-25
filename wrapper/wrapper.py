import torch.nn as nn

from utils.onnx_utils import export_onnx, parse_onnx_model
from IR.executor import IRExecutor
from adapter.factory import create_adapter
from utils.trace import trace_instant, trace_block


class ONNXModelWrapper(nn.Module):
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
        trace_instant("Converting ONNX IR to backend IR")
        self.ir_graph = parse_onnx_model(onnx_model)

        # 创建 adapter
        trace_instant(f"Creating {backend} backend adapter")
        self.adapter = create_adapter(backend)

        trace_instant(f"Creating backend IR executor")
        self.executor = IRExecutor(self.ir_graph, self.adapter, backend_type=backend)

    def forward(self, x):
        # 默认拷贝回cpu用于展示
        return self.adapter.to_numpy(self.executor.run({"input": x}))
