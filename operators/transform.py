from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("Flatten")
class FlattenOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        Flatten算子
        将输入张量展平为二维或多维形式
        """
        x = tensors[self.inputs[0]]

        # ONNX Flatten 默认 axis 为 1
        axis = getattr(self, "axis", 1)
        if hasattr(self, 'attributes'):
            for attr in self.attributes:
                if attr.name == "axis":
                    axis = attr.i

        # 调用 adapter 的 flatten 方法
        y = adapter.flatten(x, axis)
        tensors[self.outputs[0]] = y