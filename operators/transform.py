import numpy as np
from adapter.base import BackendAdapter
from .base import BackendOp
from .registry import register_operator


@register_operator("Transpose")
class TransposeOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        perm = []
        for attr in self.attributes:
            if attr.name == "perm":
                perm = list(attr.ints)
                break

        if not perm:
            perm = list(reversed(range(len(x.shape))))

        y = adapter.transpose(x, perm)
        tensors[self.outputs[0]] = y


@register_operator("Squeeze")
class SqueezeOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        self.axes = None
        for attr in attributes:
            if attr.name == "axes":
                self.axes = list(attr.ints)

    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]

        if len(self.inputs) > 1:
            axes_tensor = tensors[self.inputs[1]]
            axes = adapter.to_numpy(axes_tensor).tolist()
        else:
            axes = self.axes

        y = adapter.squeeze(x, axes)
        tensors[self.outputs[0]] = y


@register_operator("Unsqueeze")
class UnsqueezeOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs, attributes)
        self.axes = []
        for attr in attributes:
            if attr.name == "axes":
                self.axes = list(attr.ints)

    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]

        # 若 axes 来自 constant
        if len(self.inputs) > 1:
            axes_tensor = adapter.to_numpy(tensors[self.inputs[1]])
            axes = [int(x) for x in np.ravel(axes_tensor)]
        else:
            axes = self.axes

        # 调用 adapter
        y = adapter.unsqueeze(x, axes)
        tensors[self.outputs[0]] = y


@register_operator("Flatten")
class FlattenOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs, attributes)
        self.axis = 1
        for attr in attributes:
            if attr.name == "axis":
                self.axis = attr.i

    def run(self, tensors, adapter: BackendAdapter):
        """
        Flatten算子
        展平tensor
        """
        x = tensors[self.inputs[0]]

        y = adapter.flatten(x, self.axis)
        tensors[self.outputs[0]] = y


@register_operator("Reshape")
class ReshapeOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        shape = adapter.to_numpy(tensors[self.inputs[1]])
        y = adapter.reshape(x, shape)
        tensors[self.outputs[0]] = y


@register_operator("Expand")
class ExpandOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        shape = adapter.to_numpy(tensors[self.inputs[1]])
        tensors[self.outputs[0]] = adapter.expand(x, shape)


@register_operator("Slice")
class SliceOp(BackendOp):
    def run(self, tensors, adapter: BackendAdapter):
        x = tensors[self.inputs[0]]
        starts = adapter.to_numpy(tensors[self.inputs[1]]).tolist()
        ends = adapter.to_numpy(tensors[self.inputs[2]]).tolist()

        axes = None
        steps = None
        if len(self.inputs) > 3:
            axes = adapter.to_numpy(tensors[self.inputs[3]]).tolist()
        if len(self.inputs) > 4:
            steps = adapter.to_numpy(tensors[self.inputs[4]]).tolist()

        y = adapter.slice(x, starts, ends, axes, steps)
        tensors[self.outputs[0]] = y


@register_operator("Concat")
class ConcatOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)

        self.axis = None
        for attr in attributes:
            if attr.name == "axis":
                self.axis = int(attr.i)
                break

        if self.axis is None:
            raise ValueError("Concat node must have 'axis' attribute")

    def run(self, tensors, adapter: BackendAdapter):
        # 收集输入 tensor
        xs = [tensors[name] for name in self.inputs]

        if len(xs) == 0:
            raise RuntimeError("Concat requires at least one input")

        # rank 校验
        rank = len(xs[0].shape)
        for x in xs:
            if len(x.shape) != rank:
                raise RuntimeError("Concat inputs must have same rank")

        # 处理负 axis
        axis = self.axis
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise RuntimeError(f"Concat axis out of range: {self.axis}")

        # 调用 backend
        y = adapter.concat(xs, axis)

        tensors[self.outputs[0]] = y
