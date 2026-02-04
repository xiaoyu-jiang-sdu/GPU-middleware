import numpy as np
from onnx import numpy_helper

from adapter import BackendAdapter
from operators import register_operator, BackendOp


@register_operator("Constant")
class ConstantOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        self.value = None
        for attr in attributes:
            if attr.name == "value":
                # attr.t 是 ONNX TensorProt
                # 转换为 np
                self.value = numpy_helper.to_array(attr.t).astype(np.float32)
                break
        if self.value is None:
            raise ValueError("Constant node must have 'value' attribute")

    def run(self, tensors, adapter: BackendAdapter):
        """
        常量算子，读取attr中的值
        """
        tensors[self.outputs[0]] = adapter.tensor(self.value)


@register_operator("ConstantOfShape")
class ConstantOfShapeOp(BackendOp):
    def __init__(self, name, inputs, outputs, attributes):
        super().__init__(name, inputs, outputs)
        self.value = 0.0
        for attr in attributes:
            if attr.name == "value":
                self.value = float(np.array(attr.t).reshape(-1)[0])

    def run(self, tensors, adapter: BackendAdapter):
        shape = adapter.to_numpy(tensors[self.outputs[0]]).astype(int)
        out = np.full(shape, self.value, dtype=np.float32)
        tensors[self.outputs[0]] = adapter.tensor(out)
