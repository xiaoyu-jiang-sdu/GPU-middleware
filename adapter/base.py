from abc import ABC, abstractmethod

"""
后端硬件抽象接口
该抽象类定义了所有计算后端（CPU / GPU / MLU / NPU）
推理阶段提供统一接口
Operator 层只允许通过本接口调用底层计算资源，
实现模型逻辑与具体硬件平台的解耦。
"""


class BackendAdapter(ABC):
    # ================= 张量 =================
    @abstractmethod
    def tensor(self, data):
        # 创建张量
        pass

    @abstractmethod
    def to_numpy(self, tensor):
        # 张量拷贝转换为numpy
        pass

    # ================= 矩阵运算 =================
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

    # ================= 卷积 =================
    @abstractmethod
    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        # 2d卷积
        pass

    # ================= 池化 =================
    @abstractmethod
    def global_avg_pool(self, x):
        # 全局平均池化
        pass

    # ================= 张量变换 =================
    @abstractmethod
    def transpose(self, x):
        # 转置
        pass

    @abstractmethod
    def mul_scalar(self, x, scalar: float):
        # 张量与标量相乘
        pass

    @abstractmethod
    def max_pool2d(self, x, kernel_size, stride, padding):
        pass

    @abstractmethod
    def flatten(self, x, axis):
        pass
