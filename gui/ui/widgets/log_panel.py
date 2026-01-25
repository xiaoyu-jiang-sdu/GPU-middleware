from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QTextEdit
)


class LogPanelWidget(QGroupBox):
    def __init__(self):
        super().__init__("运行日志")

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text)

    def log(self, msg: str):
        self.text.append(msg)