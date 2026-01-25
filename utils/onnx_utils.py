import torch
import onnx
import tempfile
from IR.graph import IRGraph, IRNode
from onnx import numpy_helper
from utils.trace import trace


@trace("Exporting onnx IR (nn.Module -> onnx IR)")
def export_onnx(model: torch.nn.Module, input_shape):
    """
    将 torch.Module 导出为 ONNX，并返回 onnx.ModelProto
    """
    model.eval()

    dummy_input = torch.randn(*input_shape)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )
    return onnx.load(onnx_path)


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
