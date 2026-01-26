from abc import ABC, abstractmethod

from adapter.factory import wrap_adapter_method

"""
后端硬件抽象接口
该抽象类定义了所有计算后端（CPU / GPU / MLU / NPU）
推理阶段提供统一接口
Operator 层只允许通过本接口调用底层计算资源，
实现模型逻辑与具体硬件平台的解耦。
"""


class BackendAdapter(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for attr_name, attr in list(cls.__dict__.items()):
            # 只处理 callable
            if not callable(attr):
                continue
            if attr_name.startswith("_"):
                continue

            # 跳过抽象方法
            if getattr(attr, "__isabstractmethod__", False):
                continue

            # 包装方法
            wrapped = wrap_adapter_method(attr_name, attr)
            setattr(cls, attr_name, wrapped)

    # =========================
    # 张量管理
    # =========================
    @abstractmethod
    def tensor(self, data, cache, cache_key):
        # 创建张量
        pass

    @abstractmethod
    def to_numpy(self, tensor):
        # 张量拷贝转换为numpy
        pass

    # =========================
    # 基础算子
    # =========================
    @abstractmethod
    def add(self, a, b):
        # 逐元素加法
        pass

    @abstractmethod
    def matmul(self, a, b):
        # 矩阵乘
        pass

    @abstractmethod
    def relu(self, x):
        pass

    @abstractmethod
    def transpose(self, x):
        # 转置
        pass

    @abstractmethod
    def mul_scalar(self, x, scalar: float):
        # 张量与标量相乘
        pass

    # =========================
    # CNN算子
    # =========================
    @abstractmethod
    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        # 2d卷积
        pass

    @abstractmethod
    def global_avg_pool(self, x):
        # 全局平均池化
        pass

    @abstractmethod
    def max_pool2d(self, x, kernel_size, stride, padding):
        pass

    @abstractmethod
    def flatten(self, x, axis):
        pass

    # =========================
    # 归一化
    # =========================
    @abstractmethod
    def batch_norm_2d(self, x, weight, bias, running_mean, running_var, eps):
        pass
