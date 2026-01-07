from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("Conv")
class ConvOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        self.stride = (1, 1)
        self.padding = (0, 0)
        for attr in attributes:
            if attr.name == "strides":
                self.stride = tuple(attr.ints)
            elif attr.name == "pads":
                # ONNX pads: [pad_top, pad_left, pad_bottom, pad_right]
                # 取 (pad_top, pad_left)
                self.padding = (attr.ints[0], attr.ints[1])

    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]  # 输入特征图
        w = tensors[self.inputs[1]]  # 卷积权重
        b = tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # bias 可选

        # 调用 adapter 的 conv2d 方法
        y = adapter.conv2d(x, w, b, stride=self.stride, padding=self.padding)

        tensors[self.outputs[0]] = y