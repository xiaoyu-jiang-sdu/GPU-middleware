from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("BatchNormalization")
class BatchNormalizationOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)

        # 默认参数
        self.epsilon = 1e-5
        self.momentum = 0.9
        self.training_mode = 0

        for attr in attributes:
            if attr.name == "epsilon":
                self.epsilon = attr.f
            elif attr.name == "momentum":
                self.momentum = attr.f
            elif attr.name == "training_mode":
                self.training_mode = attr.i

    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        scale = tensors[self.inputs[1]]
        bias = tensors[self.inputs[2]]
        running_mean = tensors[self.inputs[3]]
        running_var = tensors[self.inputs[4]]

        y = adapter.batch_norm_2d(
            x,
            weight=scale,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
            eps=self.epsilon
        )

        tensors[self.outputs[0]] = y
