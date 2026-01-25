import matplotlib.pyplot as plt
import networkx as nx

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class GraphViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def show_architecture(self, backend):
        self.ax.clear()
        G = nx.DiGraph()
        nodes = ["PyTorch", "ONNX", "IR", "Executor", backend]
        G.add_edges_from(zip(nodes[:-1], nodes[1:]))

        pos = nx.spring_layout(G, seed=2026)
        nx.draw(G, pos, ax=self.ax, with_labels=True,
                node_size=2500, node_color="#BBDEFB")

        self.ax.set_title("系统架构视图")
        self.ax.axis("off")
        self.canvas.draw_idle()