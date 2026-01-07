from typing import Dict, Type
from .base import BackendOp

# 存放算子类型名 -> BackendOp 子类的映射
OPERATOR_REGISTRY: Dict[str, Type[BackendOp]] = {}


# 注解形式注册算子
def register_operator(op_type: str):
    def decorator(cls):
        OPERATOR_REGISTRY[op_type] = cls
        return cls
    return decorator


# 反射方式对operator实例化
def create_backend_op(ir_node):
    cls = OPERATOR_REGISTRY.get(ir_node.op_type)
    if cls is None:
        raise NotImplementedError(f"{ir_node.op_type} not supported")
    return cls(
        name=ir_node.op_type,
        inputs=ir_node.inputs,
        outputs=ir_node.outputs,
        attributes=ir_node.attributes
    )
