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


@register_operator("Add")
class AddOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        元素级加法算子
        对两个输入张量逐元素相加: y = a + b
        """
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.add(a, b)


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

        if self.transA:
            A = adapter.transpose(A)
        if self.transB:
            B = adapter.transpose(B)

        Y = adapter.matmul(A, B)
        Y = adapter.mul_scalar(Y, self.alpha)

        if C is not None:
            Y = adapter.add(Y, adapter.mul_scalar(C, self.beta))

        tensors[self.outputs[0]] = Y


@register_operator("Identity")
class IdentityOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        恒等算子
        将输入直接拷贝到输出，不做任何计算
        """
        x = tensors[self.inputs[0]]
        tensors[self.outputs[0]] = x