import sys
import time
import torch
import networkx as nx
import matplotlib

# Matplotlib 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 固定torch随机数
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QTextEdit, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from model.model import ONNXAwareModel

# ==========================
# 异步推理线程
# ==========================
class InferenceThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(torch.Tensor, float)

    def __init__(self, model, input_tensor, backend):
        super().__init__()
        self.model = model
        self.input_tensor = input_tensor
        self.backend = backend

    def run(self):
        start_time = time.time()
        self.log_signal.emit(f"开始在 {self.backend} 执行推理...")
        out = torch.tensor(self.model(self.input_tensor))  # 确保输出是Tensor
        elapsed_ms = (time.time() - start_time) * 1000
        self.log_signal.emit(f"推理完成，用时 {elapsed_ms:.2f} ms")
        self.done_signal.emit(out, elapsed_ms)

# ==========================
# GUI 原型系统
# ==========================
class PrototypeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("跨硬件推理原型系统（Prototype）")
        self.resize(1200, 800)
        self.model = None
        self.ir_graph = None
        self.input_tensor = None
        self.current_backend = "CUDA"
        self.thread = None

        self._init_ui()
        self.visualize_architecture()  # 启动时直接显示架构图

    # -------------------
    # UI 初始化
    # -------------------
    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # ===== 控制栏 =====
        control_box = QGroupBox("控制栏")
        control_layout = QHBoxLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["CUDA", "CPU"])
        control_layout.addWidget(QLabel("目标硬件:"))
        control_layout.addWidget(self.backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18", "ResNet50"])
        control_layout.addWidget(QLabel("模型:"))
        control_layout.addWidget(self.model_combo)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["系统架构视图", "关键推理流程", "IR节点前10"])
        self.view_combo.currentIndexChanged.connect(self.update_view)
        control_layout.addWidget(QLabel("可视化模式:"))
        control_layout.addWidget(self.view_combo)

        self.load_input_btn = QPushButton("生成输入")
        self.load_input_btn.clicked.connect(self.load_input)
        control_layout.addWidget(self.load_input_btn)

        self.run_btn = QPushButton("运行推理")
        self.run_btn.clicked.connect(self.run_inference)
        control_layout.addWidget(self.run_btn)

        control_box.setLayout(control_layout)
        main_layout.addWidget(control_box)

        # ===== 可视化区 =====
        vis_layout = QHBoxLayout()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        vis_layout.addWidget(self.canvas)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        vis_layout.addWidget(self.output_text)

        main_layout.addLayout(vis_layout)

        # ===== 性能区 =====
        main_layout.addWidget(QLabel("性能指标"))
        self.perf_text = QTextEdit()
        self.perf_text.setReadOnly(True)
        self.perf_text.setFixedHeight(100)
        main_layout.addWidget(self.perf_text)

    # -------------------
    # 输入生成
    # -------------------
    def load_input(self):
        self.input_tensor = torch.randn(1, 3, 224, 224)
        self.output_text.append("已生成随机输入 Tensor")

    # -------------------
    # 推理执行（异步）
    # -------------------
    def run_inference(self):
        backend = self.backend_combo.currentText()
        model_name = self.model_combo.currentText()
        self.current_backend = backend

        self.output_text.append(f"\n=== 运行推理 ===")
        self.output_text.append(f"硬件后端: {backend}")
        self.output_text.append(f"模型: {model_name}")

        # 创建模型
        base_model = self.create_model(model_name)
        self.model = ONNXAwareModel(
            base_model,
            input_shape=(1, 3, 224, 224),
            backend=backend.lower()
        )

        self.ir_graph = self.model.ir_graph

        # 可视化当前视图
        self.update_view()

        # -------------------
        # 启动异步推理线程
        # -------------------
        self.thread = InferenceThread(self.model, self.input_tensor, backend)
        self.thread.log_signal.connect(lambda msg: self.output_text.append(msg))
        self.thread.done_signal.connect(self.show_inference_result)
        self.thread.start()

    # -------------------
    # 显示推理结果
    # -------------------
    def show_inference_result(self, out, elapsed_ms):
        self.output_text.append(f"输出 shape: {tuple(out.shape)}")
        self.output_text.append(f"输出 Tensor:\n{out}")
        self.perf_text.setText(f"推理时间: {elapsed_ms:.2f} ms\n执行后端: {self.current_backend}")

    # -------------------
    # 视图切换
    # -------------------
    def update_view(self):
        mode = self.view_combo.currentText()
        if mode == "系统架构视图":
            self.visualize_architecture()
        elif mode == "关键推理流程":
            self.visualize_key_steps()
        else:
            self.visualize_ir_nodes()

    # -------------------
    # 系统架构图（规整布局）
    # -------------------
    def visualize_architecture(self):
        self.ax.clear()
        G = nx.DiGraph()

        nodes = [
            "PyTorch Model",
            "ONNX Export",
            "IR Builder",
            "IR Executor",
            "Backend Adapter",
            self.current_backend
        ]
        G.add_nodes_from(nodes)
        G.add_edges_from(zip(nodes[:-1], nodes[1:]))
        pos = nx.spring_layout(G, seed=2026)  # seed固定布局

        nx.draw(
            G, pos,
            ax=self.ax,
            with_labels=True,
            node_size=2600,
            node_color="#BBDEFB",
            font_size=10,
            arrows=True
        )
        self.ax.set_title("系统整体架构视图")
        self.ax.axis("off")
        self.canvas.draw_idle()

    # -------------------
    # 关键推理流程图
    # -------------------
    def visualize_key_steps(self):
        self.ax.clear()
        G = nx.DiGraph()

        steps = [
            "Input Tensor",
            "Conv / BN / ReLU",
            "Residual Add",
            "Global Avg Pool",
            "Fully Connected",
            "Output"
        ]
        G.add_nodes_from(steps)
        G.add_edges_from(zip(steps[:-1], steps[1:]))
        pos = nx.spring_layout(G, seed=2026)  # seed固定布局
        nx.draw(
            G, pos,
            ax=self.ax,
            with_labels=True,
            node_size=2400,
            node_color="#C8E6C9",
            font_size=10,
            arrows=True
        )
        self.ax.set_title("关键推理流程（摘要视图）")
        self.ax.axis("off")
        self.canvas.draw_idle()

    # -------------------
    # IR节点前10展示
    # -------------------
    def visualize_ir_nodes(self):
        self.ax.clear()
        G = nx.DiGraph()

        if not self.ir_graph or not self.ir_graph.nodes:
            self.ax.text(0.5, 0.5, "IR Graph为空", ha='center', va='center')
            self.canvas.draw_idle()
            return

        # 取前10个非Identity节点
        ir_nodes = [node for node in self.ir_graph.nodes if node.op_type != "Identity"][:10]

        output_to_node = {}
        for node in ir_nodes:
            for out in node.outputs:
                output_to_node[out] = node.name

        # 添加节点和边
        for node in ir_nodes:
            G.add_node(node.name)
            for inp in node.inputs:
                if inp in output_to_node:
                    G.add_edge(output_to_node[inp], node.name)

        # 使用 spring_layout 布局
        pos = nx.spring_layout(G, seed=2026)

        # 绘图
        nx.draw(
            G, pos,
            ax=self.ax,
            with_labels=True,
            node_size=1200,
            node_color="#FFCC80",
            font_size=8,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=12
        )

        self.ax.set_title("IR Graph 前10节点展示")
        self.ax.axis("off")
        self.canvas.draw_idle()

    # -------------------
    # 模型工厂
    # -------------------
    def create_model(self, model_name):
        import torchvision.models as models
        if model_name == "ResNet18":
            return models.resnet18(num_classes=10)
        return models.resnet50(num_classes=10)


# ==========================
# 启动 GUI
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PrototypeGUI()
    gui.show()
    sys.exit(app.exec_())
