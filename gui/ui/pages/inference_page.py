from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout
)

from config.project_config import ProjectConfig
from gui.ui.widgets.control_bar import ControlBarWidget
from gui.ui.widgets.log_panel import LogPanelWidget
from gui.ui.widgets.perf_panel import PerfPanelWidget

# 推理线程（你已有）
from gui.threads.inference_thread import InferenceThread
from gui.ui.widgets.trace_gantt import TraceGanttWidget
from gui.utils.device_utils import load_devices


class InferencePage(QWidget):
    def __init__(self):
        super().__init__()

        self.control_bar = ControlBarWidget()
        self.log_panel = LogPanelWidget()
        self.perf_panel = PerfPanelWidget()
        self.trace_view = TraceGanttWidget()

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.control_bar)

        center_layout = QHBoxLayout()
        center_layout.addWidget(self.log_panel, stretch=2)
        center_layout.addWidget(self.trace_view, stretch=3)

        main_layout.addLayout(center_layout)
        main_layout.addWidget(self.perf_panel)
        self._thread = None

        # 加载设备
        self.devices = []
        self.devices = load_devices()
        # 动态设置后端下拉选项
        device_names = self.devices.keys()
        if device_names:
            self.control_bar.backend_combo.clear()
            self.control_bar.backend_combo.addItems(device_names)

        # 信号连接
        self.control_bar.run_clicked.connect(self.on_run_clicked)

    # 运行推理
    def on_run_clicked(self, device_name: str, model_name: str):
        self.log_panel.log(f"启动推理：设备={device_name}, 模型={model_name}")

        if device_name not in self.devices:
            self.log_panel.log(f"错误：未找到设备 {device_name}")
            return

        device = self.devices[device_name]
        # 创建推理线程
        self._thread = InferenceThread(
            model=model_name,
            device=device
        )

        # 连接日志与完成信号
        self._thread.log_signal.connect(self.log_panel.log)
        self._thread.done_signal.connect(self.on_inference_done)

        # 启动线程
        self._thread.start()

    # 推理完成回调
    def on_inference_done(self, _, elapsed_ms: float):
        self.log_panel.log(f"推理完成")

        backend = self.control_bar.backend_combo.currentText()
        model = self.control_bar.model_combo.currentText()
        trace_path = ProjectConfig.trace_dir() / "trace.json"
        self.trace_view.load_trace(trace_path, model, backend)

        self.perf_panel.update_perf(
            backend=self.control_bar.backend_combo.currentText(),
            time_ms=elapsed_ms
        )
