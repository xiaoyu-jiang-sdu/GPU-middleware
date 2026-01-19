from typing import Dict, Type
from .base import BackendAdapter

"""
Factory 模式
通过注册，传入adapter的名称创建特定类型的后端
使用**kwargs 传入所需参数
"""

# adapter注册表
ADAPTER_REGISTRY: Dict[str, Type[BackendAdapter]] = {}


def register_adapter(name: str):
    def decorator(cls):
        ADAPTER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_adapter(name: str, **kwargs) -> BackendAdapter:
    name = name.lower()

    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY.keys())
        raise RuntimeError(
            f"Adapter '{name}' is not available.\n"
            f"Available adapters: [{available}]"
        )

    cls = ADAPTER_REGISTRY[name]
    return cls(**kwargs)
