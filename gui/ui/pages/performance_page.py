from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class PerformancePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.perf_text = QTextEdit()
        self.perf_text.setReadOnly(True)
        layout.addWidget(self.perf_text)

    def update_performance(self, elapsed_ms: float, backend: str):
        self.perf_text.setText(f"推理时间: {elapsed_ms:.2f} ms\n执行后端: {backend}")
