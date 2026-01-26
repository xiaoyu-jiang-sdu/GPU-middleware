"""
后端算子类
一切实际运行的算子都继承自此类
"""
import ast
import inspect
import textwrap
from abc import abstractmethod, ABC


# 自动收集子类中对adapter方法的调用
def _collect_adapter_calls(func, adapter_name="adapter"):
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    calls = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == adapter_name:
                        calls.add(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


class BackendOp(ABC):

    def __init__(self, name, inputs, outputs, attributes=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.ADAPTER_METHODS = set()

        run = cls.__dict__.get("run")
        if run is None:
            return

        if getattr(run, "__isabstractmethod__", False):
            return

        # 参数名校验, 强制run方法传入adapter
        sig = inspect.signature(run)
        if "adapter" not in sig.parameters:
            raise RuntimeError(
                f"{cls.__name__}.run 必须使用 adapter 作为参数名"
            )

        try:
            cls.ADAPTER_METHODS = _collect_adapter_calls(run)
        except Exception:
            cls.ADAPTER_METHODS = set()

    @abstractmethod
    def run(self, tensors, adapter):
        pass
