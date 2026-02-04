from typing import Union, Tuple, List

import torch
import onnx
from IR.graph import IRGraph, IRNode
from onnx import numpy_helper
from utils.trace import trace
import io


@trace("Exporting onnx IR (nn.Module -> onnx IR)")
def export_onnx(
    model: torch.nn.Module,
    dummy_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    input_names: List[str] = None,
    output_names=None,
    opset_version: int = 14,
) -> onnx.ModelProto:
    """
    将 torch.Module 导出为 ONNX，并返回 onnx.ModelProto
    """
    model.eval()

    # 统一处理 dummy_inputs 为 tuple
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = (dummy_inputs,)
    elif not isinstance(dummy_inputs, (tuple, list)):
        raise TypeError("dummy_inputs must be Tensor or tuple/list of Tensors")

    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(dummy_inputs))]
    if len(input_names) != len(dummy_inputs):
        raise ValueError(f"input_names length ({len(input_names)}) != dummy_inputs count ({len(dummy_inputs)})")

    # 内存导出
    buffer = io.BytesIO()

    torch.onnx.export(
        model,
        dummy_inputs,
        buffer,
        opset_version=opset_version,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names
    )

    buffer.seek(0)
    onnx_model = onnx.load(buffer)
    node_types = set()

    for node in onnx_model.graph.node:
        node_types.add(node.op_type)

    print(f"Total unique node types: {len(node_types)}")
    print("Node types:", sorted(node_types))
    return onnx_model


@trace("Parsing backend IR (onnx IR -> backend IR)")
def parse_onnx_model(onnx_model):
    graph = onnx_model.graph
    ir = IRGraph()

    ir.inputs = [i.name for i in graph.input]
    ir.outputs = [o.name for o in graph.output]

    for init in graph.initializer:
        ir.initializers[init.name] = numpy_helper.to_array(init)

    for node in graph.node:
        ir.add_node(IRNode(
            node.op_type,
            node.input,
            node.output,
            node.attribute
        ))

    return ir
