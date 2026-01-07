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
    # 根据name生成适配器实例
    cls = ADAPTER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Adapter for '{name}' not found")
    return cls(**kwargs)
