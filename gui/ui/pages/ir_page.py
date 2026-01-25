from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class IRPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def visualize_ir(self, ir_graph):
        self.ax.clear()
        G = nx.DiGraph()
        nodes = [node.name for node in ir_graph.nodes]
        G.add_nodes_from(nodes)
        edges = []
        for node in ir_graph.nodes:
            for inp in node.inputs:
                edges.append((inp, node.name))
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=2026)
        nx.draw(G, pos, ax=self.ax, with_labels=True, node_color="#FFCC80")
        self.ax.set_title("IR Graph 可视化")
        self.canvas.draw_idle()
