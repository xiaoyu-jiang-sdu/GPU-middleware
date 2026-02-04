from adapter import BackendAdapter
from operators import register_operator, BackendOp


@register_operator("Cast")
class CastOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        self.to = None
        for attr in attributes:
            if attr.name == "to":
                self.to = attr.i

    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        tensors[self.outputs[0]] = adapter.cast(x, self.to)
