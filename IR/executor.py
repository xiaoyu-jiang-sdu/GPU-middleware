from operators.registry import create_backend_op


class IRExecutor:
    def __init__(self, ir_graph, adapter):
        self.adapter = adapter
        self.ops = [create_backend_op(n) for n in ir_graph.nodes]
        self.tensors = {}

        # 加载权重
        # 根据特定后端加载到不同位置
        for name, value in ir_graph.initializers.items():
            self.tensors[name] = adapter.tensor(value)

    def set_inputs(self, inputs: dict):
        for k, v in inputs.items():
            self.tensors[k] = self.adapter.tensor(v)

    def run(self):
        for op in self.ops:
            op.run(self.tensors, self.adapter)

        return self.tensors