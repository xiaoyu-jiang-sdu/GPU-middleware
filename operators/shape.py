from adapter import BackendAdapter
from operators import register_operator, BackendOp


@register_operator("Shape")
class ShapeOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        tensors[self.outputs[0]] = adapter.shape(x)
