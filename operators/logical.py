from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("Equal")
class EqualOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        a = tensors[self.inputs[0]]
        b = tensors[self.inputs[1]]
        tensors[self.outputs[0]] = adapter.equal(a, b)


@register_operator("Where")
class WhereOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        cond = tensors[self.inputs[0]]
        x = tensors[self.inputs[1]]
        y = tensors[self.inputs[2]]
        tensors[self.outputs[0]] = adapter.where(cond, x, y)
