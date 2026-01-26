from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QSizePolicy, QGroupBox
)
from PyQt5.QtGui import QFont

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx

from models.registry import MODEL_REGISTRY, build_model
from wrapper.wrapper import ONNXModelWrapper

matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class IRPage(QWidget):
    def __init__(self):
        super().__init__()

        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Microsoft YaHei", 16))
        self.model_combo.setObjectName("ControlCombo")
        self._load_models()

        self.run_btn = QPushButton("▶ 生成")
        self.run_btn.setObjectName("RunButton")

        model_label = QLabel("模型")
        model_label.setObjectName("ControlLabel")
        model_label.setFont(QFont("Microsoft YaHei", 16))

        # 外壳 GroupBox
        top_group = QGroupBox()
        top_group.setObjectName("ControlBar")  # QSS选择器
        top_group.setTitle("")
        top_group.setFixedHeight(70)

        top_layout = QHBoxLayout(top_group)
        top_layout.setContentsMargins(16, 8, 16, 8)
        top_layout.setSpacing(16)
        top_layout.addWidget(model_label)
        top_layout.addWidget(self.model_combo)
        top_layout.addStretch()
        top_layout.addWidget(self.run_btn)
        # IR DAG
        self.ir_group = QGroupBox("IR 前 10 个节点")
        ir_layout = QVBoxLayout(self.ir_group)
        ir_layout.setContentsMargins(12, 16, 12, 12)

        self.graph_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_canvas.figure.tight_layout(rect=[0, 0, 1, 0.95])

        ir_layout.addWidget(self.graph_canvas)

        # Op → Adapter 映射
        self.mapping_group = QGroupBox("Op → Adapter 方法映射")
        map_layout = QVBoxLayout(self.mapping_group)
        map_layout.setContentsMargins(12, 16, 12, 12)

        self.mapping_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.mapping_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mapping_canvas.figure.tight_layout(rect=[0, 0, 1, 0.95])

        map_layout.addWidget(self.mapping_canvas)

        # 总布局
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(top_group)
        layout.addWidget(self.ir_group, stretch=1)
        layout.addWidget(self.mapping_group, stretch=1)

        self.run_btn.clicked.connect(self._on_run_clicked)

    def _load_models(self):
        self.model_combo.clear()
        self.model_combo.addItems(sorted(MODEL_REGISTRY.keys()))

    def _on_run_clicked(self):
        model_name = self.model_combo.currentText()
        entry = MODEL_REGISTRY.get(model_name)
        if not entry:
            return

        base_model, input_shape = build_model(
            model_name=model_name,
            num_classes=getattr(entry, "num_classes", 1000)
        )

        wrapper = ONNXModelWrapper(base_model, input_shape)

        self._draw_ir_graph(wrapper.ir_graph)
        self._draw_op_adapter_mapping(wrapper.op_adapter_mapping())

    def _draw_ir_graph(self, ir_graph):
        G = nx.DiGraph()
        nodes = [n for n in ir_graph.nodes if n.op_type != "Identity"][:10]

        for n in nodes:
            G.add_node(n.name, label=n.op_type)

        for n in nodes:
            for out in n.outputs:
                for s in nodes:
                    if out in s.inputs:
                        G.add_edge(n.name, s.name)

        fig = self.graph_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")

        pos = nx.spring_layout(G, k=1.2, seed=2026)
        nx.draw(
            G, pos,
            labels=nx.get_node_attributes(G, "label"),
            node_color="#DBEAFE",
            edge_color="#94A3B8",
            node_size=1200,
            font_size=9,
            ax=ax,
            arrows=True
        )

        self.graph_canvas.draw_idle()

    def _draw_op_adapter_mapping(self, mapping):
        G = nx.DiGraph()

        op_nodes = []
        method_nodes = []
        edge_list = []

        for op_name, info in mapping.items():
            if op_name not in op_nodes:
                op_nodes.append(op_name)
                G.add_node(op_name, ntype="op")

            for method in info.get("adapter_methods", []):
                if method not in method_nodes:
                    method_nodes.append(method)
                    G.add_node(method, ntype="method")
                edge_list.append((op_name, method))

        G.add_edges_from(edge_list)

        fig = self.mapping_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")

        x_center = 0.5
        x_offset = 0.35
        top_margin = 0.9
        bottom_margin = 0.1

        if len(op_nodes) > 1:
            y_gap = (top_margin - bottom_margin) / (len(op_nodes) - 1)
        else:
            y_gap = 0

        pos = {}
        for i, op in enumerate(op_nodes):
            pos[op] = (x_center, top_margin - i * y_gap)

        for i, method in enumerate(method_nodes):
            side = -1 if i % 2 == 0 else 1
            if len(method_nodes) > 1:
                y_method_gap = (top_margin - bottom_margin) / (len(method_nodes) - 1)
            else:
                y_method_gap = 0
            pos[method] = (x_center + side * x_offset, top_margin - i * y_method_gap)

        nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle="->", arrowsize=16, width=1.2)

        for n, (x, y) in pos.items():
            if G.nodes[n]["ntype"] == "op":
                text_color = "white"
                bbox_color = "#60A5FA"
                edge_color = "#1E40AF"
                boxstyle = "round,pad=0.4"
            else:
                text_color = "black"
                bbox_color = "#A7F3D0"
                edge_color = "#065F46"
                boxstyle = "round,pad=0.3"

            ax.text(
                x, y, n,
                fontsize=10,
                ha="center", va="center",
                color=text_color,
                bbox=dict(
                    facecolor=bbox_color,
                    edgecolor=edge_color,
                    boxstyle=boxstyle,
                    linewidth=1.5
                )
            )

        self.mapping_canvas.draw_idle()
