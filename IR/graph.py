class IRNode:
    _id_counter = 0

    def __init__(self, op_type, inputs, outputs, attributes=None, group=None):
        self.id = IRNode._id_counter
        IRNode._id_counter += 1

        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or []

        self.name = f"{op_type}_{self.id}"


class IRGraph:
    def __init__(self):
        self.nodes = []  # 节点
        self.inputs = []  # 输入
        self.outputs = []  # 输出

    def add_node(self, node: IRNode):
        self.nodes.append(node)
