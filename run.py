import numpy as np
from IR.onnx_parser import parse_onnx
from IR.executor import IRExecutor
from adapter.factory import create_adapter


def main():
    # 1. 解析 ONNX
    ir_graph = parse_onnx("onnx/resnet50.onnx")

    # 2. 选择后端
    adapter = create_adapter("cuda", device="cuda:0")
    # adapter = create_adapter("cuda")

    # 3. 构建执行器
    executor = IRExecutor(ir_graph, adapter)

    # 4. 构造输入
    input_name = ir_graph.inputs[0]

    # ResNet-50 输入尺寸: batch_size=1, channels=3, H=224, W=224
    x = np.random.randn(2, 3, 224, 224).astype(np.float32)

    executor.set_inputs({input_name: x})

    # 5. 执行
    outputs = executor.run()

    # 6. 获取输出
    for name in ir_graph.outputs:
        print(f"{name} =", adapter.to_numpy(outputs[name]))


if __name__ == "__main__":
    main()