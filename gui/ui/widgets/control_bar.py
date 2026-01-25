from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtGui import QFont

from models.registry import MODEL_REGISTRY


class ControlBarWidget(QGroupBox):
    run_clicked = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        # 总体设计
        self.setObjectName("ControlBar")
        self.setTitle("")
        self.setFixedHeight(70)

        # 下拉框
        self.backend_combo = QComboBox()
        self.backend_combo.setObjectName("ControlCombo")

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ControlCombo")
        self._load_models()

        # 推理按钮
        self.run_btn = QPushButton("▶ 运行推理")
        self.run_btn.setObjectName("RunButton")
        self.run_btn.setFixedWidth(140)

        # 标签
        backend_label = QLabel("后端")
        backend_label.setObjectName("ControlLabel")
        backend_label.setFont(QFont("Microsoft YaHei", 16))

        model_label = QLabel("模型")
        model_label.setObjectName("ControlLabel")
        model_label.setFont(QFont("Microsoft YaHei", 16))

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        layout.addWidget(backend_label)
        layout.addWidget(self.backend_combo)
        layout.addSpacing(20)
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addStretch()
        layout.addWidget(self.run_btn)

        self.run_btn.clicked.connect(self._emit_run)

    def _load_models(self):
        """动态加载注册的模型名"""
        self.model_combo.clear()
        model_names = sorted(MODEL_REGISTRY.keys())
        self.model_combo.addItems(model_names)

    def _emit_run(self):
        self.run_clicked.emit(
            self.backend_combo.currentText(),
            self.model_combo.currentText()
        )
