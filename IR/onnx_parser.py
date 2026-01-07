import onnx
from .graph import IRGraph, IRNode
import numpy as np

def parse_onnx(onnx_path: str) -> IRGraph:
    model = onnx.load(onnx_path)
    graph = model.graph

    ir_graph = IRGraph()
    ir_graph.inputs = [i.name for i in graph.input]
    ir_graph.outputs = [o.name for o in graph.output]

    # 解析权重
    # 权重暂存在cpu上作为中间表示
    for init in graph.initializer:
        ir_graph.initializers[init.name] = np.frombuffer(
            init.raw_data, dtype=np.float32
        ).reshape(init.dims)

    for node in graph.node:
        ir_graph.add_node(IRNode(
            node.op_type,
            node.input,
            node.output,
            node.attribute
        ))

    return ir_graph