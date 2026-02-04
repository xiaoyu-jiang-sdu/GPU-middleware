from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


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


@register_operator("Sub")
class SubOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.sub(a, b)


@register_operator("Mul")
class MulOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.mul(a, b)


@register_operator("Div")
class DivOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.div(a, b)


@register_operator("Pow")
class PowOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.pow(a, b)


@register_operator("Mod")
class ModOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.mod(a, b)


@register_operator("Identity")
class IdentityOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        """
        恒等算子
        将输入直接拷贝到输出
        """
        x = tensors[self.inputs[0]]
        tensors[self.outputs[0]] = x
