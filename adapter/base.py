from abc import ABC, abstractmethod

"""
后端硬件抽象接口
该抽象类定义了所有计算后端（CPU / GPU / MLU / NPU）
推理阶段提供统一接口
Operator 层只允许通过本接口调用底层计算资源，
实现模型逻辑与具体硬件平台的解耦。
"""


class BackendAdapter(ABC):
    # =========================
    # 张量管理
    # =========================
    @abstractmethod
    def tensor(self, data, cache=False, cache_key=None):
        # 创建张量
        pass

    @abstractmethod
    def to_numpy(self, tensor):
        # 张量拷贝转换为numpy
        pass

    # =========================
    # 基础算子, 二元运算符
    # =========================
    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def sub(self, a, b):
        pass

    @abstractmethod
    def mul(self, a, b):
        pass

    @abstractmethod
    def div(self, a, b):
        pass

    @abstractmethod
    def pow(self, a, b):
        pass

    @abstractmethod
    def mod(self, a, b):
        pass

    @abstractmethod
    def mul_scalar(self, x, scalar: float):
        # 张量与标量相乘
        pass

    # =========================
    # logical 算子
    # =========================
    @abstractmethod
    def equal(self, a, b):
        pass

    @abstractmethod
    def where(self, cond, x, y):
        pass

    # =========================
    # NN 算子
    # =========================
    @abstractmethod
    def matmul(self, a, b, alpha=1.0, beta=0.0, transA=False, transB=False, C=None):
        # 矩阵乘
        pass

    @abstractmethod
    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        # 2d卷积
        pass

    # =========================
    # 激活算子
    # =========================
    @abstractmethod
    def relu(self, x):
        pass

    @abstractmethod
    def erf(self, x):
        pass

    @abstractmethod
    def sqrt(self, x):
        pass
    # =========================
    # 池化算子
    # =========================
    @abstractmethod
    def global_avg_pool(self, x):
        # 全局平均池化
        pass

    @abstractmethod
    def max_pool2d(self, x, kernel_size, stride, padding):
        pass

    # =========================
    # 归一化
    # =========================
    @abstractmethod
    def batch_norm_2d(self, x, weight, bias, running_mean, running_var, eps):
        pass

    # =========================
    # shape view
    # =========================
    @abstractmethod
    def transpose(self, x, perm):
        # 转置
        pass

    @abstractmethod
    def unsqueeze(self, x, axes):
        # 再x的shape中加入维数
        pass

    @abstractmethod
    def reshape(self, x, shape):
        pass

    # =========================
    # transform
    # =========================
    @abstractmethod
    def flatten(self, x, axis):
        pass

    @abstractmethod
    def concat(self, xs, axis):
        # 将xs按照axis进行拼接
        pass

    # =========================
    # type_cast
    # =========================
    @abstractmethod
    def cast(self, x, to):
        pass

    def squeeze(self, x, axes):
        pass

    def expand(self, x, shape):
        pass

    def slice(self, x, starts, ends, axes, steps):
        pass

    def shape(self, x):
        pass
