from operators.registry import create_backend_op


class IRExecutor:
    def __init__(self, ir_graph, adapter, backend_type:str):
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
            op.run(self.tensors, self.adapter)
            for name, t in self.tensors.items():
                if hasattr(self.adapter, "to_numpy"):
                    arr = self.adapter.to_numpy(t)
                    print(f"[{self.backend_type}] after {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")

        return self.tensors[self.graph.outputs[0]]