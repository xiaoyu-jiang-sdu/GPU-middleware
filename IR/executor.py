from operators.registry import create_backend_op
from utils.trace import trace_block_emit


class IRExecutor:
    def __init__(self, ir_graph, adapter, backend_type: str):
        self.adapter = adapter
        self.ops = [create_backend_op(n) for n in ir_graph.nodes]
        self.graph = ir_graph
        self.tensors = {}
        self.backend_type = backend_type.upper()

        for name, np_array in ir_graph.initializers.items():
            self.tensors[name] = adapter.tensor(
                np_array,
                cache=True,
                cache_key=name
            )

    def run(self, inputs: dict):
        for k, v in inputs.items():
            self.tensors[k] = v

        for op in self.ops:
            with trace_block_emit("Executing ops:", op=op.name):
                self.adapter._current_op = op.name
                op.run(self.tensors, self.adapter)
        return self.tensors[self.graph.outputs[0]]

    def collect_op_adapter_mapping(self):
        """
        统计当前 IR graph 中使用到的 op 类型
        以及它们对应使用的 adapter 方法
        """
        mapping = {}

        for op in self.ops:
            op_cls = op.__class__
            op_name = op_cls.__name__

            if op_name not in mapping:
                mapping[op_name] = {
                    "adapter_methods": set(op_cls.ADAPTER_METHODS),
                    "count": 0,
                }

            mapping[op_name]["count"] += 1

        return mapping
