class IRNode:
    def __init__(self, op_type, inputs, outputs, attributes=None):
        self.op_type = op_type
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attributes = attributes or []


class IRGraph:
    def __init__(self):
        self.nodes = [] # 节点
        self.inputs = [] # 输入
        self.outputs = [] # 输出
        self.initializers = {} # 权重

    def add_node(self, node: IRNode):
        self.nodes.append(node)