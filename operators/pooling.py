from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("MaxPool")
class MaxPoolOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        # 默认参数
        self.kernel_shape = (2, 2)
        self.stride = (2, 2)
        self.padding = (0, 0)

        # 解析 ONNX 属性
        for attr in attributes:
            if attr.name == "kernel_shape":
                self.kernel_shape = tuple(attr.ints)
            elif attr.name == "strides":
                self.stride = tuple(attr.ints)
            elif attr.name == "pads":
                # ONNX pads: [pad_top, pad_left, pad_bottom, pad_right]
                self.padding = (attr.ints[0], attr.ints[1])

    def run(self, tensors, adapter: BackendAdapter):
        """
        MaxPool算子
        对输入张量执行二维最大池化操作
        """
        x = tensors[self.inputs[0]]
        # 调用 adapter 的 max_pool2d 方法
        y = adapter.max_pool2d(
            x,
            kernel_size=self.kernel_shape,
            stride=self.stride,
            padding=self.padding
        )
        tensors[self.outputs[0]] = y


@register_operator("GlobalAveragePool")
class GlobalAveragePoolOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        GlobalAveragePool算子
        对输入张量在 H、W 维度上进行全局平均池化
        """
        x = tensors[self.inputs[0]]
        y = adapter.global_avg_pool(x)
        tensors[self.outputs[0]] = y
