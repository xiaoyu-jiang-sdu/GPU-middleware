from operators.registry import create_backend_op


class IRExecutor:
    def __init__(self, ir_graph, adapter):
        self.adapter = adapter
        self.ops = [create_backend_op(n) for n in ir_graph.nodes]
        self.graph = ir_graph
        self.tensors = {}

    def run(self, inputs: dict, parameters: dict):
        self.tensors.clear()
        for k, v in inputs.items():
            self.tensors[k] = v
        # 注入 torch Parameter
        for name, param in parameters.items():
            self.tensors[name] = param
        # 顺序执行 IR
        for op in self.ops:
            op.run(self.tensors, self.adapter)
        return self.tensors[self.graph.outputs[0]]