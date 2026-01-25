from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QLabel
)


class PerfPanelWidget(QGroupBox):
    def __init__(self):
        super().__init__("性能指标")

        self.label = QLabel("尚未运行")
        self.label.setStyleSheet("font-weight: bold;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def update_perf(self, backend, time_ms):
        self.label.setText(
            f"Backend: {backend} | Latency: {time_ms:.2f} ms"
        )
