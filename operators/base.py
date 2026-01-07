"""
后端算子类
一切实际运行的算子都继承自此类
"""
from abc import abstractmethod, ABC


class BackendOp(ABC):
    def __init__(self, name, inputs, outputs, attributes=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or []

    @abstractmethod
    def run(self, tensors, adapter):
        # 执行算子，通过 Adapter 调用后端
        pass
