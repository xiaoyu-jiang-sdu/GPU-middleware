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
        b = tensors[self.inputs[2]] if len(self.inputs) > 2 else None  # bias

        # 调用 adapter 的 conv2d 方法
        y = adapter.conv2d(x, w, b, stride=self.stride, padding=self.padding)

        tensors[self.outputs[0]] = y


@register_operator("Gemm")
class GemmOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)

        # ONNX Gemm默认属性
        self.alpha = 1.0
        self.beta = 1.0
        self.transA = 0
        self.transB = 0

        for attr in attributes:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "beta":
                self.beta = attr.f
            elif attr.name == "transA":
                self.transA = attr.i
            elif attr.name == "transB":
                self.transB = attr.i

    def run(self, tensors, adapter: BackendAdapter):
        """
        全连接（Gemm）算子
        执行 Y = alpha * (A @ B) + beta * C
        支持矩阵转置选项 transA 和 transB
        """
        A = tensors[self.inputs[0]]
        B = tensors[self.inputs[1]]
        C = tensors[self.inputs[2]] if len(self.inputs) > 2 else None

        Y = adapter.matmul(A, B,
                           alpha=self.alpha,
                           beta=self.beta,
                           transA=self.transA,
                           transB=self.transB,
                           C=C)
        tensors[self.outputs[0]] = Y


@register_operator("MatMul")
class MatMulOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        矩阵乘法算子
        对两个输入矩阵执行矩阵乘法: y = a @ b
        """
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.matmul(a, b)

