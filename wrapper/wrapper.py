import torch
import torch.nn as nn

from utils.onnx_utils import export_onnx, parse_onnx_model
from IR.executor import IRExecutor
from adapter.factory import create_adapter
from utils.trace import trace_instant


class ONNXModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, dummy_inputs, backend="cuda"):
        """
        model: 原始 torch.nn.Module
        dummy_inputs: torch.Tensor 或 tuple/list of torch.Tensor
        backend: cuda / mlu / npu
        """
        super().__init__()
        self.model = model
        self.backend = backend

        # 统一处理 dummy_inputs
        if isinstance(dummy_inputs, torch.Tensor):
            dummy_inputs_tuple = (dummy_inputs,)
        else:
            dummy_inputs_tuple = tuple(dummy_inputs)

        onnx_model = export_onnx(
            model,
            dummy_inputs_tuple,
            input_names=[f"input_{i}" for i in range(len(dummy_inputs_tuple))],
            output_names=None,
        )

        # 构建 IR
        trace_instant("Converting ONNX IR to backend IR")
        self.ir_graph = parse_onnx_model(onnx_model)

        # 创建 adapter
        trace_instant(f"Creating {backend.upper()} backend adapter")
        self.adapter = create_adapter(backend)

        trace_instant(f"Creating backend IR executor")
        self.executor = IRExecutor(self.ir_graph, self.adapter, backend_type=backend)

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            inputs = tuple(args[0])
        else:
            inputs = args
        expected_count = len(self.ir_graph.inputs)
        if len(inputs) != expected_count:
            raise ValueError(f"Expected {expected_count} inputs，but actually got {len(inputs)} 个")

        input_dict = {
            name: self.adapter.tensor(tensor)
            for name, tensor in zip(self.ir_graph.inputs, inputs)
        }
        # 执行 backend IR
        output = self.executor.run(input_dict)
        # 默认拷贝回cpu用于展示
        return self.adapter.to_numpy(output)

    def op_adapter_mapping(self):
        # 返回使用的op与adapter方法映射
        return self.executor.collect_op_adapter_mapping()
