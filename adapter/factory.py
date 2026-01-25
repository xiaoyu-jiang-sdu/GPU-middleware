import functools
from collections import defaultdict

"""
Factory 模式
通过注册，传入adapter的名称创建特定类型的后端
使用**kwargs 传入所需参数
"""

# adapter注册表
ADAPTER_REGISTRY = {}

# op实际调用的adapter方法
ADAPTER_CALL_TRACE = defaultdict(set)


def register_adapter(name: str):
    def decorator(cls):
        ADAPTER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_adapter(name: str, **kwargs):
    name = name.lower()

    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY.keys())
        raise RuntimeError(
            f"Adapter '{name}' is not available.\n"
            f"Available adapters: [{available}]"
        )

    cls = ADAPTER_REGISTRY[name]
    return cls(**kwargs)


# 追踪op调用的方法
def wrap_adapter_method(method_name, fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # 调用但是不存在op
        # 一定是executor
        op_name = getattr(self, "_current_op", "IRExecutor")
        ADAPTER_CALL_TRACE[op_name].add(method_name)
        return fn(self, *args, **kwargs)
    return wrapper
