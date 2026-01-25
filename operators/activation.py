from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("Relu")
class ReluOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        Relu激活算子
        对输入张量逐元素执行 ReLU: y = max(0, x)
        """
        x = tensors[self.inputs[0]]
        tensors[self.outputs[0]] = adapter.relu(x)
